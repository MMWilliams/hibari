"""hibari G2P vs the reference pipeline (misaki + spaCy + espeak): phoneme error rate (character edits
over reference length) and sentence exact match on dump_misaki.py output, plus timing.

    python scripts/g2p/eval_g2p.py g2p_work/eval.jsonl [--no-normalize] [--show 20]
"""
import argparse, collections, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default="g2p_work/eval.jsonl")
ap.add_argument("--no-normalize", action="store_true")
ap.add_argument("--show", type=int, default=12)
a = ap.parse_args()

t0 = time.perf_counter()
from hibari.g2p import G2P  # noqa: E402
g2p = G2P()
g2p.oov  # load the OOV net too
load_ms = (time.perf_counter() - t0) * 1000


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]


rows = [json.loads(l) for l in open(a.path, encoding="utf-8")]
n = ok = 0
err = tot = 0
err_oov = tot_oov = n_oov = ok_oov = 0
word_diffs = collections.Counter()
examples = []
dt = 0.0
for r in rows:
    t1 = time.perf_counter()
    ps = g2p(r["t"], normalize_text=not a.no_normalize)
    dt += time.perf_counter() - t1
    ref = r["ps"]
    e = edits(ref, ps) if ps != ref else 0
    n += 1; ok += ps == ref; err += e; tot += len(ref)
    if r["oov"]:
        n_oov += 1; ok_oov += ps == ref; err_oov += e; tot_oov += len(ref)
    if ps != ref:
        a_, b_ = ps.split(" "), ref.split(" ")
        if len(a_) == len(b_):
            for x, y in zip(a_, b_):
                if x != y:
                    word_diffs[(x, y)] += 1
        else:
            word_diffs[("<token count>", f"{len(a_)} vs {len(b_)}")] += 1
        if len(examples) < a.show:
            examples.append((r["t"], ps, ref))
print(f"{a.path}: {n} sentences  PER {100 * err / tot:.2f}%  exact {100 * ok / n:.1f}%  "
      f"| sentences with espeak-fallback words: {n_oov}, PER {100 * err_oov / max(tot_oov, 1):.2f}%, exact {100 * ok_oov / max(n_oov, 1):.1f}%  "
      f"| without: PER {100 * (err - err_oov) / max(tot - tot_oov, 1):.2f}%, exact {100 * (ok - ok_oov) / max(n - n_oov, 1):.1f}%")
print(f"load {load_ms:.0f} ms; {dt / n * 1000:.3f} ms per sentence")
print("word diffs (ours, ref):", word_diffs.most_common(40))
for t, p, ref in examples:
    print("T:", t); print("  ours:", p); print("  ref: ", ref)
