"""Label a word list with misaki's espeak-ng fallback (the reference for out-of-vocabulary words):
word<TAB>phonemes, one per line. Training data for hibari's OOV net; espeak itself is not shipped.

    python scripts/g2p/espeak_words.py --words g2p_work/words_lex.txt --out g2p_work/espeak_lex.tsv
"""
import argparse, sys
from misaki import espeak
from misaki.token import MToken

ap = argparse.ArgumentParser()
ap.add_argument("--words", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--british", action="store_true")
a = ap.parse_args()
fb = espeak.EspeakFallback(british=a.british)
words = [w.rstrip("\n") for w in open(a.words, encoding="utf-8") if w.strip()]
with open(a.out, "w", encoding="utf-8") as f:
    for i, w in enumerate(words):
        try:
            ps, _ = fb(MToken(text=w, tag="", whitespace=""))
        except Exception as e:
            print("ERR", repr(w), e, file=sys.stderr); continue
        if ps:
            f.write(f"{w}\t{ps}\n")
        if i % 20000 == 0:
            print(i, len(words), flush=True)
print("done")
