"""End to end: the same hibari model fed by hibari's G2P vs by the reference G2P (misaki + spaCy +
espeak), on unseen sentences; UTMOS (SpeechMOS utmos22_strong) and WER (Whisper base), as in
scripts/compare_paradee.py. Sentences: lines[30000:] of the shuffled corpus, none used in training.

    python scripts/g2p/eval_e2e.py --n 50 --out runs/g2p_e2e
"""
import argparse, gzip, json, os, random, re, sys, time
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched"))
from hibari import Hibari  # noqa: E402

SR = 24000
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", default="runs/g2p_e2e")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)

from misaki import en, espeak  # noqa: E402
ref = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
tts = Hibari()
items = []
for text in lines[30000:]:
    _, toks = ref(text)
    ps_ref = "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in toks).strip()
    ps_ours = tts.phonemize(text)
    if len(ps_ours) != 1:
        continue
    items.append((text, ps_ref, ps_ours[0]))
    if len(items) == a.n:
        break
same = sum(p == q for _, p, q in items)
print(f"{len(items)} sentences; identical phoneme strings: {same}")

T = {"misaki_g2p": 0.0, "hibari_g2p": 0.0}
for i, (text, ps_ref, ps_ours) in enumerate(items):
    for k, ps in (("misaki_g2p", ps_ref), ("hibari_g2p", ps_ours)):
        t0 = time.perf_counter()
        y = tts.from_phonemes(ps, seed=0)
        T[k] += time.perf_counter() - t0
        sf.write(f"{a.out}/{i:02d}_{k}.wav", y, SR)

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


rows = {}
for k in T:
    mos, err, nw = [], 0, 0
    for i, (text, _, _) in enumerate(items):
        y, _ = sf.read(f"{a.out}/{i:02d}_{k}.wav", dtype="float32")
        with torch.no_grad():
            mos.append(float(utmos(torch.from_numpy(y)[None].cuda(), SR)))
        hyp = asr({"raw": y, "sampling_rate": SR})["text"]
        r = norm(text); err += edits(r, norm(hyp)); nw += len(r)
    rows[k] = {"utmos": float(np.mean(mos)), "wer": err / nw, "synth_s": T[k]}
    print(f"{k:12s} UTMOS {rows[k]['utmos']:.3f}  WER {100 * rows[k]['wer']:.1f}%")
diff = [(t, p, q) for t, p, q in items if p != q]
json.dump({"n": len(items), "identical": same, "rows": rows, "diffs": diff}, open(f"{a.out}/results.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
for t, p, q in diff[:10]:
    print("T:", t); print("  misaki:", p); print("  hibari:", q)
