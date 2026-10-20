"""Check the lexicon/G2P port in isolation: feed the reference pipeline's own spaCy tokens and tags
(and an oracle for its espeak fallback) through hibari's logic and compare phoneme strings.
Also checks hibari's tokenizer against spaCy's tokens.

    python scripts/g2p/check_port.py g2p_work/dev.jsonl
"""
import collections, gzip, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.g2p import G2P, Tok, _load_json  # noqa: E402
from hibari.g2p.lexicon import Lexicon  # noqa: E402
from hibari.g2p.tokenize import Tokenizer  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "g2p_work/dev.jsonl", encoding="utf-8")]
g = G2P.__new__(G2P)
g.lexicon = Lexicon(_load_json("lexicon_us_gold.json.gz"), _load_json("lexicon_us_silver.json.gz"))
g.unk = ""
oracle = {}
g._oov = lambda w: oracle.get(w)

n = ok = 0
diffs = collections.Counter()
examples = []
for r in rows:
    oracle = {w: ps for w, ps in r["oov"]}
    g._oov = (lambda w, o=oracle: o.get(w))
    toks = [Tok(t, tag, ws) for t, tag, ws in r["tok"]]
    out = g.phonemize_tokens(toks)
    ps = "".join((tk.phonemes or "") + (" " if tk.whitespace else "") for tk in out).strip()
    n += 1
    if ps == r["ps"]:
        ok += 1
    else:
        a, b = ps.split(" "), r["ps"].split(" ")
        for x, y in zip(a, b):
            if x != y:
                diffs[(x, y)] += 1
        if len(examples) < 15:
            examples.append((r["t"], ps, r["ps"]))
print(f"lexicon/core port with reference tokens+tags+oracle OOV: {ok}/{n} sentences exact ({ok / n:.4f})")
print("word diffs (ours, ref):", diffs.most_common(30))
for t, a, b in examples[:8]:
    print("T:", t); print("  ours:", a); print("  ref: ", b)

tok = Tokenizer(_load_json("tokenizer_exceptions.json"))
n = ok = 0
bad = collections.Counter()
ex2 = []
for r in rows:
    ref = [t for t, _, _ in r["tok"]]
    ours = [t for t, _ in tok(r["t"])]
    n += 1
    if ours == ref:
        ok += 1
    else:
        s1, s2 = set(ours), set(ref)
        for x in s1 - s2:
            bad[("ours", x)] += 1
        for x in s2 - s1:
            bad[("ref", x)] += 1
        if len(ex2) < 10:
            ex2.append((r["t"], ours, ref))
print(f"\ntokenizer vs spaCy: {ok}/{n} sentences identical ({ok / n:.4f})")
print("token diffs:", bad.most_common(40))
for t, a, b in ex2[:6]:
    print("T:", t); print("  ours:", a); print("  ref: ", b)
