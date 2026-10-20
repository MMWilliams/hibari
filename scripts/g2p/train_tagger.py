"""Train hibari's averaged-perceptron tagger on spaCy's tags (g2p_work/train*.jsonl from dump_misaki.py).

    python scripts/g2p/train_tagger.py --limit 300000 --epochs 5 --out src/hibari/assets/tagger.json.gz
"""
import argparse, collections, glob, gzip, json, os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.g2p.tagger import features, context, norm, Tagger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="g2p_work/train*.jsonl")
ap.add_argument("--dev", default="g2p_work/dev.jsonl")
ap.add_argument("--limit", type=int, default=300000)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--prune", type=float, default=0.5, help="drop averaged weights below this magnitude")
ap.add_argument("--minfreq", type=int, default=10, help="words seen fewer times than this share one feature per shape")
ap.add_argument("--out", default="src/hibari/assets/tagger.json.gz")
a = ap.parse_args()


def load(paths, limit=None):
    sents = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            toks = json.loads(line)["tok"]
            if toks:
                sents.append(([t for t, _, _ in toks], [g for _, g, _ in toks]))
            if limit and len(sents) >= limit:
                return sents
    return sents


train = load(sorted(glob.glob(a.train)), a.limit)
dev = load([a.dev])
print(f"train {len(train)} sentences, dev {len(dev)}", flush=True)

# tag dictionary: frequent words with one dominant tag (case-sensitive)
counts = collections.defaultdict(collections.Counter)
for words, tags in train:
    for w, t in zip(words, tags):
        counts[w][t] += 1
tagdict = {}
for w, c in counts.items():
    tag, n = c.most_common(1)[0]
    tot = sum(c.values())
    if tot >= 10 and n / tot >= 0.97:
        tagdict[w] = tag
classes = sorted({t for _, tags in train for t in tags})
wfreq = collections.Counter()
for words, _ in train:
    for w in words:
        wfreq[norm(w)] += 1
vocab = {w for w, c in wfreq.items() if c >= a.minfreq}
print(f"tagdict {len(tagdict)} words, {len(classes)} tags, feature vocabulary {len(vocab)} words", flush=True)

weights, totals, tstamps = {}, {}, {}
step = 0


def predict(feats):
    scores = collections.defaultdict(float)
    for f in feats:
        ws = weights.get(f)
        if ws:
            for t, w in ws.items():
                scores[t] += w
    return max(classes, key=lambda t: (scores[t], t))


def update(truth, guess, feats):
    global step
    step += 1
    if truth == guess:
        return
    for f in feats:
        ws = weights.setdefault(f, {})
        for t, v in ((truth, 1.0), (guess, -1.0)):
            k = (f, t)
            totals[k] = totals.get(k, 0.0) + (step - tstamps.get(k, 0)) * ws.get(t, 0.0)
            tstamps[k] = step
            ws[t] = ws.get(t, 0.0) + v


for epoch in range(a.epochs):
    random.Random(epoch).shuffle(train)
    t0, n, c = time.time(), 0, 0
    for words, tags in train:
        ctx = context(words, vocab)
        prev, prev2 = "-START-", "-START2-"
        for i, (w, truth) in enumerate(zip(words, tags)):
            guess = tagdict.get(w)
            if guess is None:
                feats = features(i, w, ctx, prev, prev2)
                guess = predict(feats)
                update(truth, guess, feats)
            n += 1
            c += guess == truth
            prev2, prev = prev, guess
    print(f"epoch {epoch}: train acc {c / n:.4f}  {time.time() - t0:.0f}s  features {len(weights)}", flush=True)

# average, prune, round
model_w = {}
for f, ws in weights.items():
    out = {}
    for t, w in ws.items():
        k = (f, t)
        avg = (totals.get(k, 0.0) + (step - tstamps.get(k, 0)) * w) / step
        if abs(avg) >= a.prune:
            out[t] = round(avg, 1)
    if out:
        model_w[f] = out
model = {"tags": classes, "tagdict": tagdict, "vocab": sorted(vocab), "weights": model_w}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with gzip.open(a.out, "wt", encoding="utf-8", compresslevel=9) as f:
    json.dump(model, f, ensure_ascii=False, separators=(",", ":"))
print(f"saved {a.out}: {os.path.getsize(a.out) / 1e6:.2f} MB, {len(model_w)} features, "
      f"{sum(len(v) for v in model_w.values())} weights", flush=True)

tagger = Tagger(model)
n = c = 0
conf = collections.Counter()
t0 = time.time()
for words, tags in dev:
    pred = tagger(words)
    for w, p, t in zip(words, pred, tags):
        n += 1
        c += p == t
        if p != t:
            conf[(w.lower(), t, p)] += 1
dt = time.time() - t0
print(f"dev token accuracy {c / n:.4f} on {n} tokens; {dt / len(dev) * 1000:.3f} ms per sentence")
print("top confusions (word, spacy, ours):", conf.most_common(40))
