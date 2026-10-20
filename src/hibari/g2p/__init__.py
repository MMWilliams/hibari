"""Text -> misaki-style English phonemes, without spaCy or espeak-ng.

    g2p = G2P()
    g2p("Hello there.")      -> 'həlˈO ðˈɛɹ.'
    G2P(british=True)("Hello there.")  -> 'həlˈQ ðˈɛː.'   (misaki's British lexicon and rules)

Pipeline (misaki 0.9.4 en.py, ported): normalize -> tokenize (spaCy rules) -> tag (perceptron)
-> lexicon with part-of-speech and context rules -> OOV net for unknown words -> stress resolution.
Phoneme characters are the ones in the model's vocab.json; the TTS layer maps them to ids."""
import gzip, json, os, re

from .token import Tok, merge_tokens
from .lexicon import (Lexicon, apply_stress, stress_weight, is_digit, PRIMARY_STRESS, VOWELS, CONSONANTS,
                      CURRENCIES, SYMBOLS)
from .tokenize import Tokenizer
from .tagger import Tagger
from .normalize import normalize

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SUBTOKEN_RE = re.compile(r"^['‘’]+|[A-ZÀ-ÖØ-Þ](?=[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ])|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}"
                         r"|[^\W\d_]*?(?:['‘’][^\W\d_])*?[a-zß-öø-ÿ](?=[A-ZÀ-ÖØ-Þ])|[^\W\d_]+(?:['‘’][^\W\d_])*"
                         r"|[^\w\-'‘’]|['‘’]+$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^\)]*)\)")
SUBTOKEN_JUNKS = frozenset("',-._‘’/")
PUNCTS = frozenset(';:,.!?—…"“”')
NON_QUOTE_PUNCTS = frozenset(p for p in PUNCTS if p not in '"“”')
PUNCT_TAGS = frozenset([".", ",", "-LRB-", "-RRB-", "``", '""', "''", ":", "$", "#", "NFP"])
PUNCT_TAG_PHONEMES = {"-LRB-": "(", "-RRB-": ")", "``": chr(8220), '""': chr(8221), "''": chr(8221)}
FALLBACK_PIECE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[,.]\d+)*|[^\w\s]+")
SYMBOL_PHONEMES = {"/": "slˈæʃ", "=": "ˈikwᵊlz", "+": "plˈʌs", "&": "ænd", "@": "æt", "%": "pəɹsˈɛnt", "#": "hˈæʃ",
                   "*": "ˈæstəɹˌɪsk", "~": "tˈɪldə", "^": "kˈɛɹət", "<": "lˈɛs ðən", ">": "ɡɹˈATəɹ ðən", "|": "", "—": "—"}


def _load_json(name):
    p = os.path.join(ASSETS, name)
    if name.endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Context:
    __slots__ = ("future_vowel", "future_to")

    def __init__(self, future_vowel=None, future_to=False):
        self.future_vowel, self.future_to = future_vowel, future_to


class G2P:
    def __init__(self, oov=None, unk="", british=False):
        """oov: callable word -> phonemes for words the lexicon can't read (default: the bundled
        OOV net, loaded lazily); pass False to skip unknown words. british: misaki's British
        lexicon and suffix rules and the British OOV net (en-gb readings)."""
        acc = "gb" if british else "us"
        self.lexicon = Lexicon(_load_json(f"lexicon_{acc}_gold.json.gz"), _load_json(f"lexicon_{acc}_silver.json.gz"),
                               british)
        self.tokenizer = Tokenizer(_load_json("tokenizer_exceptions.json"))
        self.tagger = Tagger(_load_json("tagger.json.gz"))
        self._oov = oov
        self.unk = unk
        self.british = british

    @property
    def oov(self):
        if self._oov is None:
            from .oov import OOV
            self._oov = OOV(os.path.join(ASSETS, "oov_gb.onnx" if self.british else "oov.onnx"))
        return self._oov

    # -- tokens -------------------------------------------------------------------------------
    def tokenize(self, text):
        """Tokens with tags. [text](/phonemes/) sets a token's phonemes directly."""
        toks = []
        pos = 0
        for m in LINK_RE.finditer(text):
            toks += self._tokens(text[pos:m.start()])
            f = m.group(2)
            ps = f[1:].rstrip("/") if len(f) > 1 and f[0] == "/" and f[-1] == "/" else None
            toks.append(Tok(m.group(1), "NN", " " if m.end() < len(text) and text[m.end()].isspace() else "",
                            phonemes=ps, rating=5))
            pos = m.end()
        toks += self._tokens(text[pos:])
        return toks

    def _tokens(self, text):
        pairs = self.tokenizer(text)
        if not pairs:
            return []
        tags = self.tagger([t for t, _ in pairs])
        return [Tok(t, tag, ws) for (t, ws), tag in zip(pairs, tags)]

    @staticmethod
    def retokenize(tokens):
        words = []
        currency = None
        for i, token in enumerate(tokens):
            if token.alias is None and token.phonemes is None:
                tks = [token.copy(text=t, whitespace="", is_head=True, num_flags=token.num_flags,
                                  stress=token.stress, prespace=False) for t in SUBTOKEN_RE.findall(token.text)]
            else:
                tks = [token]
            tks[-1].whitespace = token.whitespace
            for j, tk in enumerate(tks):
                if tk.alias is not None or tk.phonemes is not None:
                    pass
                elif tk.tag == "$" and tk.text in CURRENCIES:
                    currency = tk.text
                    tk.phonemes = ""
                    tk.rating = 4
                elif tk.tag == ":" and tk.text in ("-", "–"):
                    tk.phonemes = "—"
                    tk.rating = 3
                elif tk.tag in PUNCT_TAGS and not all(97 <= ord(c.lower()) <= 122 for c in tk.text):
                    tk.phonemes = PUNCT_TAG_PHONEMES.get(tk.tag, "".join(c for c in tk.text if c in PUNCTS))
                    tk.rating = 4
                elif currency is not None:
                    if tk.tag != "CD":
                        currency = None
                    elif j + 1 == len(tks) and (i + 1 == len(tokens) or tokens[i + 1].tag != "CD"):
                        tk.currency = currency
                elif 0 < j < len(tks) - 1 and tk.text == "2" and (tks[j - 1].text[-1] + tks[j + 1].text[0]).isalpha():
                    tk.alias = "to"
                if tk.alias is not None or tk.phonemes is not None:
                    words.append(tk)
                elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
                    tk.is_head = False
                    words[-1].append(tk)
                else:
                    words.append(tk if tk.whitespace else [tk])
        return [w[0] if isinstance(w, list) and len(w) == 1 else w for w in words]

    @staticmethod
    def token_context(ctx, ps, token):
        vowel = ctx.future_vowel
        if ps:
            vowel = next((None if c in NON_QUOTE_PUNCTS else (c in VOWELS) for c in ps
                          if c in VOWELS or c in CONSONANTS or c in NON_QUOTE_PUNCTS), vowel)
        future_to = token.text in ("to", "To") or (token.text == "TO" and token.tag in ("TO", "IN"))
        return Context(vowel, future_to)

    @staticmethod
    def resolve_tokens(tokens):
        text = "".join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text
        prespace = " " in text or "/" in text or len({0 if c.isalpha() else (1 if is_digit(c) else 2)
                                                        for c in text if c not in SUBTOKEN_JUNKS}) > 1
        for i, tk in enumerate(tokens):
            if tk.phonemes is None:
                if i == len(tokens) - 1 and tk.text in NON_QUOTE_PUNCTS:
                    tk.phonemes = tk.text
                    tk.rating = 3
                elif all(c in SUBTOKEN_JUNKS for c in tk.text):
                    tk.phonemes = ""
                    tk.rating = 3
            elif i > 0:
                tk.prespace = prespace
        if prespace:
            return
        indices = [(PRIMARY_STRESS in tk.phonemes, stress_weight(tk.phonemes), i) for i, tk in enumerate(tokens) if tk.phonemes]
        if len(indices) == 2 and len(tokens[indices[0][2]].text) == 1:
            i = indices[1][2]
            tokens[i].phonemes = apply_stress(tokens[i].phonemes, -0.5)
            return
        elif len(indices) < 2 or sum(b for b, _, _ in indices) <= (len(indices) + 1) // 2:
            return
        for _, _, i in sorted(indices)[:len(indices) // 2]:
            tokens[i].phonemes = apply_stress(tokens[i].phonemes, -0.5)

    def fallback(self, tk):
        """Unknown word: letters go to the OOV net, digit runs to the number reader, symbols to a
        small table (what espeak returned for them in the reference pipeline)."""
        if self._oov is False:
            return None, None
        out = []
        for m in FALLBACK_PIECE.finditer(tk.text):
            piece = m.group()
            if piece[0].isdigit():
                ps = self.lexicon.get_number(piece, None, False)[0]
            elif piece[0].isalpha():
                ps = self.oov(piece)
            else:
                ps = "".join(SYMBOL_PHONEMES.get(c, "") for c in piece)
            if ps:
                out.append(ps)
        return (" ".join(out), 2) if out else ("", 3)

    # -- main ---------------------------------------------------------------------------------
    def phonemize_tokens(self, tokens):
        """Fill in tokens' phonemes in place (misaki G2P.__call__)."""
        words = self.retokenize(tokens)
        ctx = Context()
        for w in reversed(words):
            if not isinstance(w, list):
                if w.phonemes is None:
                    w.phonemes, w.rating = self.lexicon(w, ctx)
                if w.phonemes is None:
                    w.phonemes, w.rating = self.fallback(w)
                ctx = self.token_context(ctx, w.phonemes, w)
                continue
            left, right = 0, len(w)
            should_fallback = False
            while left < right:
                if any(tk.alias is not None or tk.phonemes is not None for tk in w[left:right]):
                    tk = None
                else:
                    tk = merge_tokens(w[left:right])
                ps, rating = (None, None) if tk is None else self.lexicon(tk, ctx)
                if ps is not None:
                    w[left].phonemes = ps
                    w[left].rating = rating
                    for x in w[left + 1:right]:
                        x.phonemes = ""
                        x.rating = rating
                    ctx = self.token_context(ctx, ps, tk)
                    right = left
                    left = 0
                elif left + 1 < right:
                    left += 1
                else:
                    right -= 1
                    tk = w[right]
                    if tk.phonemes is None:
                        if all(c in SUBTOKEN_JUNKS for c in tk.text):
                            tk.phonemes = ""
                            tk.rating = 3
                        else:
                            should_fallback = True
                            break
                    left = 0
            if should_fallback:
                tk = merge_tokens(w)
                w[0].phonemes, w[0].rating = self.fallback(tk)
                for x in w[1:]:
                    x.phonemes = ""
                    x.rating = w[0].rating
            else:
                self.resolve_tokens(w)
        tokens = [merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk for tk in words]
        for tk in tokens:
            if tk.phonemes:
                tk.phonemes = tk.phonemes.replace("ɾ", "T").replace("ʔ", "t")
        return tokens

    def __call__(self, text, normalize_text=True):
        """Phoneme string for one line of text (KPipeline's join: a space where the text had one)."""
        if normalize_text:
            text = normalize(text)
        tokens = self.phonemize_tokens(self.tokenize(text))
        return "".join((self.unk if tk.phonemes is None else tk.phonemes) + (" " if tk.whitespace else "")
                       for tk in tokens).strip()
