"""Measure a rule-based G2P prototype against espeak-ng as Kokoro uses it (misaki EspeakG2P) on corpus
sentences: phoneme error rate (character edits / reference length), sentence exact match, the same on
sentences containing digits, and the most frequent word-level differences.

    python scripts/g2p_lang/eval_lang.py --lang es --espeak es --n 500
"""
import argparse, collections, importlib, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "kokoro_patched")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True, help="module in scripts/g2p_lang and data/lang/<lang>.txt")
ap.add_argument("--espeak", required=True, help="espeak-ng language code (es, it, pt-br)")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--show", type=int, default=15)
a = ap.parse_args()
proto = importlib.import_module(a.lang)
from misaki import espeak  # noqa: E402
ref_g2p = espeak.EspeakG2P(language=a.espeak)

lines = [l.strip() for l in open(os.path.join(ROOT, f"data/lang/{a.lang}.txt"), encoding="utf-8")]
random.Random(0).shuffle(lines)
lines = [l for l in lines if 4 <= len(l.split()) <= 30][: a.n]


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]


tot = {"all": [0, 0, 0, 0], "digits": [0, 0, 0, 0]}          # err, len, exact, n
diffs = collections.Counter()
examples = []
for t in lines:
    ref = ref_g2p(t)[0]
    hyp = proto.phonemize(t)
    e = edits(ref, hyp)
    for k in ("all",) + (("digits",) if any(c.isdigit() for c in t) else ()):
        tot[k][0] += e; tot[k][1] += len(ref); tot[k][2] += hyp == ref; tot[k][3] += 1
    if hyp != ref:
        rw, hw = ref.split(" "), hyp.split(" ")
        if len(rw) == len(hw):
            for x, y in zip(hw, rw):
                if x != y:
                    diffs[(x, y)] += 1
        else:
            diffs[("<tokens>", f"{len(hw)} vs {len(rw)}")] += 1
        if len(examples) < a.show:
            examples.append((t, hyp, ref))
for k, (e, n, ok, cnt) in tot.items():
    if cnt:
        print(f"{a.lang} {k:7s}: {cnt} sentences  PER {100 * e / n:.2f}%  exact {100 * ok / cnt:.1f}%")
print("top diffs (ours, espeak):", diffs.most_common(60))
for t, h, r in examples:
    print("T:", t); print("  ours:  ", h); print("  espeak:", r)
