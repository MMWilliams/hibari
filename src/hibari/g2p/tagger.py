"""Averaged-perceptron part-of-speech tagger (Penn Treebank tags), trained on spaCy en_core_web_sm's
tags so the lexicon's part-of-speech entries ('read', 'live', 'record') and function-word rules get
the same answers as the reference pipeline. Pure dict lookups: a word with one dominant tag is
answered from the tag dictionary, everything else from the weights."""

START = ["-START2-", "-START-"]
END = ["-END-", "-END2-"]


def norm(w):
    if "-" in w and w[0] != "-":
        return "!HYPHEN"
    if w.isdigit() and len(w) == 4:
        return "!YEAR"
    if w[0].isdigit():
        return "!DIGITS"
    return w.lower()


def shape(w):
    if w.isalpha():
        if w.islower():
            return "x"
        if w.isupper():
            return "XX" if len(w) > 1 else "X1"
        if w[0].isupper() and w[1:].islower():
            return "Xx"
        return "xX"
    if w.isdigit():
        return "d"
    return "".join("d" if c.isdigit() else "x" if c.isalpha() else c for c in w[:4])


def features(i, word, ctx, prev, prev2):
    """ctx = START + normalized words (rare ones already replaced by their shape) + END, so ctx[i + 2]
    is the current word."""
    j = i + 2
    return ("b", "s3=" + word[-3:], "p1=" + word[0], "w=" + ctx[j], "sh=" + shape(word), "i0" if i == 0 else "i+",
            "t-1=" + prev, "t-2=" + prev2, "t-12=" + prev + "," + prev2,
            "w-1=" + ctx[j - 1], "s3-1=" + ctx[j - 1][-3:], "w-2=" + ctx[j - 2],
            "w+1=" + ctx[j + 1], "s3+1=" + ctx[j + 1][-3:], "w+2=" + ctx[j + 2])


def context(words, vocab):
    """Normalized words for the feature window; words outside the tagger's vocabulary are replaced by
    their shape so the weight table only needs frequent words."""
    out = []
    for w in words:
        n = norm(w)
        out.append(n if vocab is None or n in vocab else "<" + shape(w) + ">")
    return START + out + END


class Tagger:
    def __init__(self, model):
        self.tagdict = model["tagdict"]
        self.tags = model["tags"]
        self.vocab = set(model["vocab"]) if "vocab" in model else None
        self.weights = model["weights"]

    def predict(self, feats):
        scores = {}
        W = self.weights
        for f in feats:
            ws = W.get(f)
            if ws:
                for t, w in ws.items():
                    scores[t] = scores.get(t, 0) + w
        return max(self.tags, key=lambda t: (scores.get(t, 0), t)) if scores else "NN"

    def __call__(self, words):
        ctx = context(words, self.vocab)
        prev, prev2 = START[1], START[0]
        tags = []
        for i, w in enumerate(words):
            tag = self.tagdict.get(w)
            if tag is None:
                tag = self.predict(features(i, w, ctx, prev, prev2))
            tags.append(tag)
            prev2, prev = prev, tag
        return tags
