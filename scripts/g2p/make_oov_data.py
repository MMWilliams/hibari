"""Collect (word, espeak phonemes) pairs for the OOV net: every fallback call recorded in the training
dumps plus the lexicon words labeled by espeak_words.py. Writes g2p_work/oov_train.tsv and
g2p_work/oov_heldout.tsv (words from the dev dump's fallback calls that are not in the train set).

--british: the same words labeled by espeak en-gb (g2p_work/gb/espeak_all.tsv from espeak_words.py
--british over every US training word; the en-us corpus fallback records are not used) and the
dev_gb.jsonl fallback calls as held-out; outputs go to g2p_work/gb/.
"""
import collections, glob, json, random, re, sys

GB = "--british" in sys.argv
D = "g2p_work/gb/" if GB else "g2p_work/"
WORD_OK = re.compile(r"^[^\W\d_][^\W\d'’\-\.]*$")
pairs = {}
seen = collections.Counter()
for p in [] if GB else sorted(glob.glob("g2p_work/train*.jsonl")):
    for line in open(p, encoding="utf-8"):
        for w, ps in json.loads(line)["oov"]:
            if ps and WORD_OK.match(w) and len(w) <= 24:
                pairs.setdefault(w, ps)
                seen[w] += 1
print("corpus OOV words", len(pairs))
lex = 0
for line in (l for p in sorted(glob.glob(D + "espeak_*.tsv")) for l in open(p, encoding="utf-8")):
    w, ps = line.rstrip("\n").split("\t")
    if WORD_OK.match(w) and len(w) <= 24 and w not in pairs:
        pairs[w] = ps
        lex += 1
print("lexicon words added", lex, "total", len(pairs))
dev = {}
for line in open("g2p_work/dev_gb.jsonl" if GB else "g2p_work/dev.jsonl", encoding="utf-8"):
    for w, ps in json.loads(line)["oov"]:
        if ps and WORD_OK.match(w) and w not in pairs:
            dev.setdefault(w, ps)
print("held-out dev OOV words (unseen)", len(dev))
chars = collections.Counter(c for ps in pairs.values() for c in ps)
print("output chars", "".join(sorted(chars)), len(chars))
bad = [w for w, ps in pairs.items() if len(ps) > 4 * len(w)]
print("too long for 4 slots per char", len(bad), bad[:20])
items = [(w, ps) for w, ps in pairs.items() if w not in bad]
random.Random(0).shuffle(items)
with open(D + "oov_train.tsv", "w", encoding="utf-8") as f:
    f.writelines(f"{w}\t{ps}\n" for w, ps in items)
with open(D + "oov_heldout.tsv", "w", encoding="utf-8") as f:
    f.writelines(f"{w}\t{ps}\n" for w, ps in dev.items())
print("corpus-OOV examples:", [(w, pairs[w]) for w, _ in seen.most_common(15)])
