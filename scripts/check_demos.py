"""Do the demo wavs play as intended? Transcribe <run>/demo*.wav with Whisper base and report the word error
rate against the sentences they were made from (<run>/demos.json); a demo that does not transcribe is a demo
that does not play. Light GPU job.

    python scripts/check_demos.py runs/voices/bf_emma [runs/voices/am_fenrir ...]
"""
import json, os, re, sys
import soundfile as sf, torch
from transformers import pipeline as hf_pipeline

asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cuda:0" if torch.cuda.is_available() else "cpu")


def norm(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]


ok = True
for run in sys.argv[1:]:
    texts = json.load(open(os.path.join(run, "demos.json"), encoding="utf-8"))
    err = n = 0
    for i, t in enumerate(texts, 1):
        y, sr = sf.read(os.path.join(run, f"demo{i}.wav"), dtype="float32")
        hyp = asr({"raw": y, "sampling_rate": sr})["text"]
        r = norm(t); e = edits(r, norm(hyp)); err += e; n += len(r)
        print(f"{run}/demo{i}.wav {len(y) / sr:5.1f} s  WER {100 * e / len(r):5.1f}%  | {hyp.strip()[:90]}")
    wer = err / n                      # Whisper spells "hibari's" its own way and writes "100" for "a hundred": ~10% is the floor
    ok &= wer < 0.25
    print(f"{run}: demos WER {100 * wer:.1f}% ({'ok' if wer < 0.25 else 'FAIL'})", flush=True)
sys.exit(0 if ok else 1)
