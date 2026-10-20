"""Record what the reference pipeline (misaki 0.9.4 + spaCy en_core_web_sm + espeak-ng fallback) does
on corpus sentences: phonemes, spaCy tokens/tags, and every espeak fallback call. Training and
reference data for hibari's own G2P (tokenizer, tagger, OOV net) and the evaluation sets.

    python scripts/g2p/dump_misaki.py --start 40000 --end 340000 --shard 0 --nshards 4 --out g2p_work/train
    python scripts/g2p/dump_misaki.py --start 31000 --end 33500 --out g2p_work/eval
"""
import argparse, gzip, json, random, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True); ap.add_argument("--end", type=int, required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--british", action="store_true", help="misaki british=True + espeak en-gb (for the British G2P port)")
a = ap.parse_args()

lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
lines = [l for l in lines if 4 <= len(l.split()) <= 35]
random.Random(0).shuffle(lines)
lines = lines[a.start:a.end][a.shard::a.nshards]

from misaki import en, espeak
fb = espeak.EspeakFallback(british=a.british)
rec = {}
class Fallback:
    def __call__(self, token):
        ps, r = fb(token)
        rec.setdefault("oov", []).append([token.text, ps])
        return ps, r
g2p = en.G2P(trf=False, british=a.british, fallback=Fallback(), unk="")
orig = g2p.tokenize
def tokenize(text, tokens, features):
    mt = orig(text, tokens, features)
    rec["tok"] = [[t.text, t.tag, t.whitespace] for t in mt]
    return mt
g2p.tokenize = tokenize

out = f"{a.out}{a.shard}.jsonl" if a.nshards > 1 else f"{a.out}.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for i, text in enumerate(lines):
        rec.clear()
        try:
            ps, tokens = g2p(text)
        except Exception as e:
            print("ERR", repr(text), e, file=sys.stderr); continue
        # the same join KPipeline.en_tokenize does (single-line text, so no chunking)
        ps = "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in tokens).strip()
        f.write(json.dumps({"t": text, "ps": ps, "tok": rec.get("tok", []), "oov": rec.get("oov", [])}, ensure_ascii=False) + "\n")
        if i % 5000 == 0:
            print(i, flush=True)
print("done", out)
