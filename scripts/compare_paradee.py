"""Head-to-head on sentences no model trained on: Kokoro-82M teacher vs Paradee-8M vs TinyKokoro.
Speed: onnxruntime, 1 thread, interleaved per sentence so CPU load hits both equally.
Quality: UTMOS (SpeechMOS utmos22_strong) and WER (Whisper base), as in the Paradee paper.

    python scripts/compare_paradee.py --tiny export/v60k --n 50
"""
import argparse, gzip, json, os, random, re, sys, time
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched")); sys.path.insert(0, os.path.join(ROOT, "export"))
from infer_numpy import Tiny  # noqa: E402

SR = 24000
ap = argparse.ArgumentParser()
ap.add_argument("--tiny", default="export/v60k")
ap.add_argument("--paradee", default="upstream/paradee/onnx")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", default="runs/compare")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# same filter + shuffle as dump_teacher.py; [30000:] was never synthesized for training
lines = [l.strip() for l in gzip.open("data/sentences.txt.gz", "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)

from kokoro import KModel, KPipeline  # noqa: E402
teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().cuda()
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=teacher)
vocab = teacher.vocab
items = []
for text in lines[30000:]:
    rs = list(pipe(text, voice="af_heart"))
    if len(rs) == 1 and rs[0].audio is not None:
        ids = [0, *[vocab[p] for p in rs[0].phonemes if p in vocab], 0]
        items.append((text, ids, rs[0].audio.cpu().numpy()))
    if len(items) == a.n:
        break
del teacher, pipe; torch.cuda.empty_cache()

import onnxruntime as ort  # noqa: E402
def session(p):
    so = ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = 1
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
par = {k: session(f"{a.paradee}/{f}") for k, f in (("paradee_int8", "paradee_int8.onnx"), ("paradee_fp32", "paradee.onnx"))}
tiny = {"tiny_int8": Tiny(f"{a.tiny}/tiny.int8.onnx"), "tiny_fp32": Tiny(f"{a.tiny}/tiny.onnx")}
synth = {**{k: (lambda ids, s=s: s.run(None, {"input_ids": np.asarray([ids], np.int64),
                                               "speed": np.ones(1, np.float32)})[0][0]) for k, s in par.items()},
         **{k: (lambda ids, t=t: t(ids)) for k, t in tiny.items()}}
sizes = {"paradee_int8": f"{a.paradee}/paradee_int8.onnx", "paradee_fp32": f"{a.paradee}/paradee.onnx",
         "tiny_int8": f"{a.tiny}/tiny.int8.onnx", "tiny_fp32": f"{a.tiny}/tiny.onnx"}
for f in synth.values():
    f(items[0][1])                                                       # warm-up
T = {k: [0.0, 0.0] for k in synth}
for i, (text, ids, wav_t) in enumerate(items):
    sf.write(f"{a.out}/{i:02d}_teacher.wav", wav_t, SR)
    for k, f in synth.items():
        t0 = time.perf_counter(); y = f(ids); T[k][0] += time.perf_counter() - t0; T[k][1] += len(y) / SR
        sf.write(f"{a.out}/{i:02d}_{k}.wav", y, SR)

# quality
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda().eval()
from transformers import pipeline as hf_pipeline  # noqa: E402
asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cuda:0")

def norm(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()

def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]

rows = []
for k in ["teacher", *synth]:
    mos, err, nw = [], 0, 0
    for i, (text, _, _) in enumerate(items):
        y, _ = sf.read(f"{a.out}/{i:02d}_{k}.wav", dtype="float32")
        with torch.no_grad():
            mos.append(float(utmos(torch.from_numpy(y)[None].cuda(), SR)))
        hyp = asr({"raw": y, "sampling_rate": SR})["text"]
        r = norm(text); err += edits(r, norm(hyp)); nw += len(r)
    rtf = T[k][0] / T[k][1] if k in T else None
    rows.append((k, os.path.getsize(sizes[k]) / 1e6 if k in sizes else None, rtf, np.mean(mos), err / nw))

print(f"\n{len(items)} unseen sentences, {sum(len(norm(t)) for t, _, _ in items)} words; 1 ORT thread, trainer running on GPU 0")
print(f"{'model':14s} {'MB':>6s} {'RTF':>7s} {'xRT':>6s} {'UTMOS':>6s} {'WER':>6s}")
for k, mb, rtf, m, w in rows:
    print(f"{k:14s} {mb if mb else 0:6.2f} {rtf if rtf else 0:7.4f} {1/rtf if rtf else 0:6.1f} {m:6.3f} {100*w:5.1f}%")
json.dump(rows, open(f"{a.out}/results.json", "w"), indent=1)

gap = np.zeros(SR // 2, np.float32)
for i in range(3):
    parts = [sf.read(f"{a.out}/{i:02d}_{k}.wav", dtype="float32")[0] for k in ("teacher", "paradee_int8", "tiny_int8")]
    sf.write(f"{a.out}/ab{i}_teacher_paradee_tiny.wav", np.concatenate([parts[0], gap, parts[1], gap, parts[2]]), SR)
