"""Per-voice evaluation on sentences no model trained on: the Kokoro-82M teacher (that voice) vs the
hibari student exported for it. The student is fed the teacher's phoneme ids, so this measures the
acoustic model alone (the G2P is evaluated separately). Speed: onnxruntime, 1 thread. Quality: UTMOS
(utmos22_strong) and WER (Whisper base), as scripts/compare_paradee.py does for af_heart.
Writes <out>/NN_{teacher,tiny_int8,tiny_fp32}.wav, results.json and abN_teacher_student.wav (3 pairs).

    python scripts/eval_voice.py --voice bf_emma --tiny export/bf_emma --n 50 --out runs/voices/bf_emma/eval
"""
import argparse, gzip, json, os, random, re, sys, time
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched")); sys.path.insert(0, os.path.join(ROOT, "export"))
from infer_numpy import Tiny  # noqa: E402

SR = 24000
ap = argparse.ArgumentParser()
ap.add_argument("--voice", required=True)
ap.add_argument("--tiny", required=True, help="export dir with tiny.onnx and tiny.int8.onnx")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# same filter + shuffle as dump_teacher.py; [30000:] was never synthesized for any voice's training
lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)

from kokoro import KModel, KPipeline  # noqa: E402
if os.environ.get("HIBARI_GPU_FRAC"):  # share of GPU memory this process may use (user limit)
    torch.cuda.set_per_process_memory_fraction(float(os.environ["HIBARI_GPU_FRAC"]))
teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().cuda()
pipe = KPipeline(lang_code=a.voice[0], repo_id="hexgrad/Kokoro-82M", model=teacher)
vocab = teacher.vocab
items = []
for text in lines[30000:]:
    rs = list(pipe(text, voice=a.voice))
    if len(rs) == 1 and rs[0].audio is not None:
        ids = [0, *[vocab[p] for p in rs[0].phonemes if p in vocab], 0]
        items.append((text, ids, rs[0].audio.cpu().numpy()))
    if len(items) == a.n:
        break
del teacher, pipe; torch.cuda.empty_cache()

tiny = {"tiny_int8": Tiny(f"{a.tiny}/tiny.int8.onnx"), "tiny_fp32": Tiny(f"{a.tiny}/tiny.onnx")}
sizes = {k: f"{a.tiny}/tiny{'.int8' if k == 'tiny_int8' else ''}.onnx" for k in tiny}
for t in tiny.values():
    t(items[0][1])                                                       # warm-up
T = {k: [0.0, 0.0] for k in tiny}
for i, (text, ids, wav_t) in enumerate(items):
    sf.write(f"{a.out}/{i:02d}_teacher.wav", wav_t, SR)
    for k, t in tiny.items():
        t0 = time.perf_counter(); y = t(ids); T[k][0] += time.perf_counter() - t0; T[k][1] += len(y) / SR
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
for k in ["teacher", *tiny]:
    mos, err, nw = [], 0, 0
    for i, (text, _, _) in enumerate(items):
        y, _ = sf.read(f"{a.out}/{i:02d}_{k}.wav", dtype="float32")
        with torch.no_grad():
            mos.append(float(utmos(torch.from_numpy(y)[None].cuda(), SR)))
        hyp = asr({"raw": y, "sampling_rate": SR})["text"]
        r = norm(text); err += edits(r, norm(hyp)); nw += len(r)
    rtf = T[k][0] / T[k][1] if k in T else None
    rows[k] = {"mb": os.path.getsize(sizes[k]) / 1e6 if k in sizes else None, "rtf": rtf,
               "x_realtime": 1 / rtf if rtf else None, "utmos": float(np.mean(mos)), "wer": err / nw}

print(f"\n{a.voice}: {len(items)} unseen sentences, {sum(len(norm(t)) for t, _, _ in items)} words; 1 ORT thread")
print(f"{'model':10s} {'MB':>6s} {'xRT':>6s} {'UTMOS':>6s} {'WER':>6s}")
for k, r in rows.items():
    print(f"{k:10s} {r['mb'] or 0:6.2f} {r['x_realtime'] or 0:6.1f} {r['utmos']:6.3f} {100 * r['wer']:5.1f}%")
json.dump({"voice": a.voice, "n": len(items), "rows": rows}, open(f"{a.out}/results.json", "w"), indent=1)

gap = np.zeros(SR // 2, np.float32)
for i in range(3):
    t_, s_ = (sf.read(f"{a.out}/{i:02d}_{k}.wav", dtype="float32")[0] for k in ("teacher", "tiny_int8"))
    sf.write(f"{os.path.dirname(a.out.rstrip('/'))}/ab{i}_teacher_student.wav", np.concatenate([t_, gap, s_]), SR)
