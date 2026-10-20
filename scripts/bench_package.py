"""Speed through the hibari package API: G2P time per sentence, synthesis x realtime (1 thread by
default), load time, on the first 20 held-out sentences of g2p_work/eval.jsonl (or any text file,
one sentence per line). Best of 3 rounds per sentence, as scripts/bench_all.py does.

    python scripts/bench_package.py [--threads 1] [--text file]
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--text")
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()

try:
    import psutil
    load0 = psutil.cpu_percent(interval=2.0)
except ImportError:
    load0 = None

t0 = time.perf_counter()
from hibari import Hibari  # noqa: E402
tts = Hibari(threads=a.threads)
tts.g2p.oov
t_load = time.perf_counter() - t0
if a.text:
    sents = [l.strip() for l in open(a.text, encoding="utf-8") if l.strip()][: a.n]
else:
    sents = [json.loads(l)["t"] for l in open(os.path.join(ROOT, "g2p_work/eval.jsonl"), encoding="utf-8")][: a.n]
tts("Warm up, twice."); tts("Warm up, twice.")

g2p_ms, best = [], [np.inf] * len(sents)
for r in range(3):
    for i, s in enumerate(sents):
        t1 = time.perf_counter()
        chunks = tts.phonemize(s)
        t2 = time.perf_counter()
        parts = [tts.from_ids(tts.ids(ps)) for ps in chunks]
        t3 = time.perf_counter()
        g2p_ms.append((t2 - t1) * 1e3)
        best[i] = min(best[i], t3 - t2)
        if r == 0:
            best.append(sum(len(p) for p in parts) / 24000) if len(best) == len(sents) else None
audio = []
for s in sents:
    audio.append(sum(len(tts.from_ids(tts.ids(ps))) for ps in tts.phonemize(s)) / 24000)
tot_audio, tot_wall = sum(audio), sum(best[: len(sents)])
load1 = psutil.cpu_percent(interval=2.0) if load0 is not None else None
print(f"hibari {os.path.getsize(tts.model_path) / 1e6:.2f} MB, {a.threads} thread(s): load {t_load:.2f} s | "
      f"{len(sents)} sentences, {tot_audio:.1f} s audio | synthesis {tot_audio / tot_wall:.0f}x realtime "
      f"(RTF {tot_wall / tot_audio:.4f}) | G2P median {np.median(g2p_ms):.2f} ms, mean {np.mean(g2p_ms):.2f} ms per sentence"
      + (f" | system CPU load before/after {load0:.0f}%/{load1:.0f}%" if load0 is not None else ""))
