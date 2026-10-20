"""Score Kokoro-82M voices themselves on unseen sentences (UTMOS + Whisper-base WER), to choose which
voices to distill. Sentences: lines[30000:] of the shuffled corpus (never used in training), one
KPipeline per language code (voice[0]: a = American, b = British).

    python scripts/score_voices.py --voices am_michael am_fenrir am_puck --n 20 --out runs/voices/score
"""
import argparse, gzip, json, os, random, re, sys
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched"))
SR = 24000
ap = argparse.ArgumentParser()
ap.add_argument("--voices", nargs="+", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--out", default="runs/voices/score")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)
texts = lines[30000: 30000 + a.n]

from kokoro import KModel, KPipeline  # noqa: E402
model = KModel(repo_id="hexgrad/Kokoro-82M").eval().cuda()
pipes = {}
wavs = {}
for v in a.voices:
    pipe = pipes.setdefault(v[0], KPipeline(lang_code=v[0], repo_id="hexgrad/Kokoro-82M", model=model))
    wavs[v] = []
    for i, t in enumerate(texts):
        y = np.concatenate([r.audio.cpu().numpy() for r in pipe(t, voice=v) if r.audio is not None])
        wavs[v].append(y)
        if i < 3:
            sf.write(f"{a.out}/{v}_{i}.wav", y, SR)

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


res = {}
for v in a.voices:
    mos, err, nw, secs = [], 0, 0, 0.0
    for t, y in zip(texts, wavs[v]):
        with torch.no_grad():
            mos.append(float(utmos(torch.from_numpy(y)[None].cuda(), SR)))
        hyp = asr({"raw": y, "sampling_rate": SR})["text"]
        r = norm(t); err += edits(r, norm(hyp)); nw += len(r); secs += len(y) / SR
    res[v] = {"utmos": float(np.mean(mos)), "utmos_min": float(np.min(mos)), "wer": err / nw, "audio_s": secs}
    print(f"{v:14s} UTMOS {res[v]['utmos']:.3f} (min {res[v]['utmos_min']:.2f})  WER {100 * res[v]['wer']:.1f}%  {secs:.0f} s", flush=True)
json.dump({"n": len(texts), "texts": texts, "voices": res}, open(f"{a.out}/results.json", "w", encoding="utf-8"), indent=1)
