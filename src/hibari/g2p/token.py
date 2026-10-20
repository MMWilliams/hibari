"""Token record shared by the tokenizer, tagger and lexicon (misaki's MToken, flattened)."""


class Tok:
    __slots__ = ("text", "tag", "whitespace", "phonemes", "is_head", "alias", "stress",
                 "currency", "num_flags", "prespace", "rating")

    def __init__(self, text, tag="", whitespace="", phonemes=None, is_head=True, alias=None, stress=None,
                 currency=None, num_flags="", prespace=False, rating=None):
        self.text, self.tag, self.whitespace, self.phonemes = text, tag, whitespace, phonemes
        self.is_head, self.alias, self.stress, self.currency = is_head, alias, stress, currency
        self.num_flags, self.prespace, self.rating = num_flags, prespace, rating

    def copy(self, **kw):
        t = Tok(*(getattr(self, s) for s in Tok.__slots__))
        for k, v in kw.items():
            setattr(t, k, v)
        return t

    def __repr__(self):
        return f"Tok({self.text!r}, {self.tag}, ws={self.whitespace!r}, ps={self.phonemes!r})"


def merge_tokens(tokens, unk=None):
    stress = {tk.stress for tk in tokens if tk.stress is not None}
    currency = {tk.currency for tk in tokens if tk.currency is not None}
    rating = {tk.rating for tk in tokens}
    if unk is None:
        phonemes = None
    else:
        phonemes = ""
        for tk in tokens:
            if tk.prespace and phonemes and not phonemes[-1].isspace() and tk.phonemes:
                phonemes += " "
            phonemes += unk if tk.phonemes is None else tk.phonemes
    return Tok(
        text="".join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text,
        tag=max(tokens, key=lambda tk: sum(1 if c == c.lower() else 2 for c in tk.text)).tag,
        whitespace=tokens[-1].whitespace,
        phonemes=phonemes,
        is_head=tokens[0].is_head,
        alias=None,
        stress=list(stress)[0] if len(stress) == 1 else None,
        currency=max(currency) if currency else None,
        num_flags="".join(sorted({c for tk in tokens for c in tk.num_flags})),
        prespace=tokens[0].prespace,
        rating=None if None in rating else min(rating),
    )
