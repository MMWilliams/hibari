"""Synthesize the distillation corpus with Kokoro-82M: audio plus the per-token durations that
produced it, so the student trains on an alignment that is correct by construction. The
teacher's own F0 and energy curves (80 Hz) are saved too, as direct prosody targets.

One shard per process: data/dist/shard{K}.{i16,f0n.f16}.bin + shard{K}.jsonl. Rows name their
own bin files, so the trainer reads all shards from one concatenated index.
"""
import _env  # noqa: F401
import argparse, json, gzip, random
from pathlib import Path
import numpy as np, torch
from kokoro import KModel, KPipeline

SPF = 600

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="data/sentences.txt.gz")
ap.add_argument("--n", type=int, default=60000)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--voice", default="af_heart")
ap.add_argument("--out", default="data/dist")
a = ap.parse_args()

lines = [l.strip() for l in gzip.open(a.text, "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)
lines = lines[: a.n][a.shard:: a.nshards]

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
binp, f0p, idxp = (out / f"shard{a.shard}{ext}" for ext in (".i16.bin", ".f0n.f16.bin", ".jsonl"))
seen = {json.loads(l)["sid"] for l in open(idxp, encoding="utf-8")} if idxp.exists() else set()

model = KModel(repo_id="hexgrad/Kokoro-82M").eval().cuda()
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
vocab = model.vocab
cap = {}
_f0n = model.predictor.F0Ntrain
def _capture(*x):
    cap["f0"], cap["n"] = r = _f0n(*x)
    return r
model.predictor.F0Ntrain = _capture

fb, ff, fi = open(binp, "ab"), open(f0p, "ab"), open(idxp, "a", encoding="utf-8")
offset, f0_off = fb.tell() // 2, ff.tell() // 2
for sid, text in enumerate(lines):
    if sid in seen:
        continue
    rows = []
    for r in pipe(text, voice=a.voice):
        if r.audio is None or r.pred_dur is None:
            continue
        ids = [0, *[vocab[p] for p in r.phonemes if p in vocab], 0]
        dur = r.pred_dur.tolist()
        wav = r.audio.cpu().numpy()
        f0 = cap["f0"].squeeze().float().cpu().numpy()
        en = cap["n"].squeeze().float().cpu().numpy()
        assert len(ids) == len(dur), (len(ids), len(dur))
        assert sum(dur) * SPF == len(wav), (sum(dur) * SPF, len(wav))
        assert len(f0) == 2 * sum(dur), (len(f0), sum(dur))
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        fb.write(pcm.tobytes())
        ff.write(np.concatenate([f0, en]).astype(np.float16).tobytes())
        rows.append({"sid": sid, "ids": ids, "dur": dur, "offset": offset, "samples": len(pcm),
                     "bin": str(binp), "f0_offset": f0_off, "f0_bin": str(f0p)})
        offset += len(pcm)
        f0_off += 2 * len(f0)
    # index rows are written after their audio, so a crash never indexes missing samples
    fb.flush(); ff.flush()
    fi.writelines(json.dumps(r) + "\n" for r in rows)
    if sid % 500 == 0:
        fi.flush()
        print(f"shard {a.shard}: {sid}/{len(lines)}", flush=True)
fb.close(); ff.close(); fi.close()
print(f"shard {a.shard} done", flush=True)
