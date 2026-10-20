"""Is Kokoro-82M itself reliable in a language? ~30 corpus sentences per language through Kokoro (its
best-graded voice, its own G2P: espeak-ng for e/f/h/i/p, misaki ja/zh), transcribed by Whisper
large-v3 (language given), scored as WER on words (space-delimited languages) and CER on characters
(all languages; the only word-free measure for ja/zh). English af_heart is scored the same way as the
reference point. Sentences come from data/lang/<lang>.txt (agentlans/high-quality-multilingual-sentences,
CC-BY-4.0), rows 5..35 words / 8..60 characters, without digits so the transcript comparison is exact.

    python scripts/lang/teacher_quality.py --n 30 --out runs/lang/teacher
"""
import argparse, json, os, random, re, sys, unicodedata
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched"))
SR = 24000
LANGS = {  # kokoro code: (voice, whisper language, corpus file, word-delimited)
    "a": ("af_heart", "english", None, True), "e": ("ef_dora", "spanish", "es", True),
    "f": ("ff_siwis", "french", "fr", True), "h": ("hf_alpha", "hindi", "hi", True),
    "i": ("if_sara", "italian", "it", True), "p": ("pf_dora", "portuguese", "pt", True),
    "j": ("jf_alpha", "japanese", "ja", False), "z": ("zf_xiaoxiao", "chinese", "zh", False)}
ap = argparse.ArgumentParser()
ap.add_argument("--langs", default="aefhipjz")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", default="runs/lang/teacher")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def sentences(code):
    _, _, f, worded = LANGS[code]
    if f is None:
        import gzip
        lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
        lines = [l for l in lines if 4 <= len(l.split()) <= 35]
        random.Random(0).shuffle(lines)
        lines = lines[30000:]
    else:
        lines = [l.strip() for l in open(os.path.join(ROOT, f"data/lang/{f}.txt"), encoding="utf-8")]
        random.Random(0).shuffle(lines)
    ok = []
    for l in lines:
        if any(c.isdigit() for c in l) or "http" in l or len(l) > 200:
            continue
        if worded and not 5 <= len(l.split()) <= 30:
            continue
        if not worded and not 8 <= len(l) <= 60:
            continue
        ok.append(l)
        if len(ok) == a.n:
            break
    return ok


def chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(c for c in s if unicodedata.category(c)[0] in "LN")


def words(s):
    s = unicodedata.normalize("NFKC", s).lower().replace("-", " ")
    return "".join(c if unicodedata.category(c)[0] in "LN" or c in "' " else " " for c in s).split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]


from kokoro import KModel, KPipeline  # noqa: E402
model = KModel(repo_id="hexgrad/Kokoro-82M").eval().cuda()
items = {}
for code in a.langs:
    voice = LANGS[code][0]
    pipe = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=model)
    rows = []
    for i, t in enumerate(sentences(code)):
        parts = [r.audio.cpu().numpy() for r in pipe(t, voice=voice) if r.audio is not None]
        if not parts:
            continue
        y = np.concatenate(parts)
        sf.write(f"{a.out}/{code}_{i:02d}.wav", y, SR)
        rows.append({"text": t, "wav": f"{a.out}/{code}_{i:02d}.wav", "sec": len(y) / SR})
    items[code] = rows
    print(f"{code} {voice}: {len(rows)} sentences, {sum(r['sec'] for r in rows):.0f} s", flush=True)
del model, pipe; torch.cuda.empty_cache()

from transformers import pipeline as hf_pipeline  # noqa: E402
asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")
res = {}
for code, rows in items.items():
    voice, wlang, _, worded = LANGS[code]
    ce = cn = we = wn = 0
    for r in rows:
        y, _ = sf.read(r["wav"], dtype="float32")
        hyp = asr({"raw": y, "sampling_rate": SR}, generate_kwargs={"language": wlang, "task": "transcribe"})["text"]
        r["hyp"] = hyp
        ce += edits(chars(r["text"]), chars(hyp)); cn += len(chars(r["text"]))
        if worded:
            we += edits(words(r["text"]), words(hyp)); wn += len(words(r["text"]))
    res[code] = {"voice": voice, "n": len(rows), "cer": ce / max(cn, 1), "wer": we / wn if worded else None,
                 "audio_s": sum(r["sec"] for r in rows)}
    print(f"{code} {voice:12s} CER {100 * res[code]['cer']:5.1f}%" + (f"  WER {100 * res[code]['wer']:5.1f}%" if worded else "")
          + f"  ({len(rows)} sentences, {res[code]['audio_s']:.0f} s)", flush=True)
json.dump({"results": res, "items": items}, open(f"{a.out}/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
