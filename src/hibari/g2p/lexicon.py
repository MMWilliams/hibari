"""misaki's English lexicon logic (misaki 0.9.4, Apache-2.0, hexgrad), ported so hibari reads text the
way Kokoro-82M was trained to hear it. Dictionary files: misaki's us_gold / us_silver, or gb_gold /
gb_silver with british=True (which also switches the three suffix rules that differ: no flapped t,
and ɪ instead of ᵻ in -es / -ed). Numbers go through numbers.py instead of num2words; everything else
follows misaki en.py."""
import re, unicodedata
from . import numbers as N

LEXICON_ORDS = frozenset([39, 45, *range(65, 91), *range(97, 123)])
CONSONANTS = frozenset("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
US_TAUS = frozenset("AIOWYiuæɑəɛɪɹʊʌ")
CURRENCIES = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
ORDINALS = frozenset(["st", "nd", "rd", "th"])
ADD_SYMBOLS = {".": "dot", "/": "slash"}
SYMBOLS = {"%": "percent", "&": "and", "+": "plus", "@": "at"}
PRIMARY_STRESS, SECONDARY_STRESS = "ˈ", "ˌ"
STRESSES = SECONDARY_STRESS + PRIMARY_STRESS
VOWELS = frozenset("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
DIPHTHONGS = frozenset("AIOQWYʤʧ")


def stress_weight(ps):
    return sum(2 if c in DIPHTHONGS else 1 for c in ps) if ps else 0


def apply_stress(ps, stress):
    def restress(ps):
        ips = list(enumerate(ps))
        stresses = {i: next(j for j, v in ips[i:] if v in VOWELS) for i, p in ips if p in STRESSES}
        for i, j in stresses.items():
            _, s = ips[i]
            ips[i] = (j - 0.5, s)
        return "".join(p for _, p in sorted(ips))
    if stress is None:
        return ps
    elif stress < -1:
        return ps.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")
    elif stress == -1 or (stress in (0, -0.5) and PRIMARY_STRESS in ps):
        return ps.replace(SECONDARY_STRESS, "").replace(PRIMARY_STRESS, SECONDARY_STRESS)
    elif stress in (0, 0.5, 1) and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(SECONDARY_STRESS + ps)
    elif stress >= 1 and PRIMARY_STRESS not in ps and SECONDARY_STRESS in ps:
        return ps.replace(SECONDARY_STRESS, PRIMARY_STRESS)
    elif stress > 1 and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(PRIMARY_STRESS + ps)
    return ps


def is_digit(text):
    return bool(re.match(r"^[0-9]+$", text))


def get_parent_tag(tag):
    if tag is None:
        return tag
    elif tag.startswith("VB"):
        return "VERB"
    elif tag.startswith("NN"):
        return "NOUN"
    elif tag.startswith("ADV") or tag.startswith("RB"):
        return "ADV"
    elif tag.startswith("ADJ") or tag.startswith("JJ"):
        return "ADJ"
    return tag


class Dict:
    """A misaki dictionary with its grow_dictionary semantics done at lookup time instead of
    load time: a lowercase key also answers to its Capitalized form and vice versa, unless the
    other form has its own entry."""

    def __init__(self, d):
        self.d = d

    def get(self, w):
        d = self.d
        v = d.get(w)
        if v is not None or len(w) < 2:
            return v
        if w == w.lower():
            c = w.capitalize()
            return d.get(c) if c != w else None
        if w == w.lower().capitalize():
            return d.get(w.lower())
        return None

    def __contains__(self, w):
        return self.get(w) is not None


class Lexicon:
    def __init__(self, gold, silver, british=False):
        self.golds, self.silvers = Dict(gold), Dict(silver)
        self.british = british
        self.cap_stresses = (0.5, 2)

    def get_NNP(self, word):
        ps = [self.golds.get(c.upper()) for c in word if c.isalpha()]
        if None in ps:
            return None, None
        ps = apply_stress("".join(ps), 0)
        ps = ps.rsplit(SECONDARY_STRESS, 1)
        return PRIMARY_STRESS.join(ps), 3

    def get_special_case(self, word, tag, stress, ctx):
        if tag == "ADD" and word in ADD_SYMBOLS:
            return self.lookup(ADD_SYMBOLS[word], None, -0.5, ctx)
        elif word in SYMBOLS:
            return self.lookup(SYMBOLS[word], None, None, ctx)
        elif "." in word.strip(".") and word.replace(".", "").isalpha() and len(max(word.split("."), key=len)) < 3:
            return self.get_NNP(word)
        elif word in ("a", "A"):
            return "ɐ" if tag == "DT" else "ˈA", 4
        elif word in ("am", "Am", "AM"):
            if tag.startswith("NN"):
                return self.get_NNP(word)
            elif ctx.future_vowel is None or word != "am" or stress and stress > 0:
                return self.golds.get("am"), 4
            return "ɐm", 4
        elif word in ("an", "An", "AN"):
            if word == "AN" and tag.startswith("NN"):
                return self.get_NNP(word)
            return "ɐn", 4
        elif word == "I" and tag == "PRP":
            return SECONDARY_STRESS + "I", 4
        elif word in ("by", "By", "BY") and get_parent_tag(tag) == "ADV":
            return "bˈI", 4
        elif word in ("to", "To") or (word == "TO" and tag in ("TO", "IN")):
            return {None: self.golds.get("to"), False: "tə", True: "tʊ"}[ctx.future_vowel], 4
        elif word in ("in", "In") or (word == "IN" and tag != "NNP"):
            stress = PRIMARY_STRESS if ctx.future_vowel is None or tag != "IN" else ""
            return stress + "ɪn", 4
        elif word in ("the", "The") or (word == "THE" and tag == "DT"):
            return "ði" if ctx.future_vowel is True else "ðə", 4
        elif tag == "IN" and re.match(r"(?i)vs\.?$", word):
            return self.lookup("versus", None, None, ctx)
        elif word in ("used", "Used", "USED"):
            if tag in ("VBD", "JJ") and ctx.future_to:
                return self.golds.get("used")["VBD"], 4
            return self.golds.get("used")["DEFAULT"], 4
        return None, None

    def is_known(self, word, tag):
        if word in self.golds or word in SYMBOLS or word in self.silvers:
            return True
        elif not word.isalpha() or not all(ord(c) in LEXICON_ORDS for c in word):
            return False
        elif len(word) == 1:
            return True
        elif word == word.upper() and word.lower() in self.golds:
            return True
        return word[1:] == word[1:].upper()

    def lookup(self, word, tag, stress, ctx):
        is_NNP = None
        if word == word.upper() and word not in self.golds:
            word = word.lower()
            is_NNP = tag == "NNP"
        ps, rating = self.golds.get(word), 4
        if ps is None and not is_NNP:
            ps, rating = self.silvers.get(word), 3
        if isinstance(ps, dict):
            if ctx and ctx.future_vowel is None and "None" in ps:
                tag = "None"
            elif tag not in ps:
                tag = get_parent_tag(tag)
            ps = ps.get(tag, ps["DEFAULT"])
        if ps is None or (is_NNP and PRIMARY_STRESS not in ps):
            ps, rating = self.get_NNP(word)
            if ps is not None:
                return ps, rating
        return apply_stress(ps, stress), rating

    def _s(self, stem):
        if not stem:
            return None
        elif stem[-1] in "ptkfθ":
            return stem + "s"
        elif stem[-1] in "szʃʒʧʤ":
            return stem + ("ɪ" if self.british else "ᵻ") + "z"
        return stem + "z"

    def stem_s(self, word, tag, stress, ctx):
        if len(word) < 3 or not word.endswith("s"):
            return None, None
        if not word.endswith("ss") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif (word.endswith("'s") or (len(word) > 4 and word.endswith("es") and not word.endswith("ies"))) and self.is_known(word[:-2], tag):
            stem = word[:-2]
        elif len(word) > 4 and word.endswith("ies") and self.is_known(word[:-3] + "y", tag):
            stem = word[:-3] + "y"
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._s(stem), rating

    def _ed(self, stem):
        if not stem:
            return None
        elif stem[-1] in "pkfθʃsʧ":
            return stem + "t"
        elif stem[-1] == "d":
            return stem + ("ɪ" if self.british else "ᵻ") + "d"
        elif stem[-1] != "t":
            return stem + "d"
        elif self.british or len(stem) < 2:
            return stem + "ɪd"
        elif stem[-2] in US_TAUS:
            return stem[:-1] + "ɾᵻd"
        return stem + "ᵻd"

    def stem_ed(self, word, tag, stress, ctx):
        if len(word) < 4 or not word.endswith("d"):
            return None, None
        if not word.endswith("dd") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif len(word) > 4 and word.endswith("ed") and not word.endswith("eed") and self.is_known(word[:-2], tag):
            stem = word[:-2]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ed(stem), rating

    def _ing(self, stem):
        if not stem:
            return None
        elif self.british:
            if stem[-1] in "əː":
                return None
        elif len(stem) > 1 and stem[-1] == "t" and stem[-2] in US_TAUS:
            return stem[:-1] + "ɾɪŋ"
        return stem + "ɪŋ"

    def stem_ing(self, word, tag, stress, ctx):
        if len(word) < 5 or not word.endswith("ing"):
            return None, None
        if len(word) > 5 and self.is_known(word[:-3], tag):
            stem = word[:-3]
        elif self.is_known(word[:-3] + "e", tag):
            stem = word[:-3] + "e"
        elif len(word) > 5 and re.search(r"([bcdgklmnprstvxz])\1ing$|cking$", word) and self.is_known(word[:-4], tag):
            stem = word[:-4]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ing(stem), rating

    def get_word(self, word, tag, stress, ctx):
        ps, rating = self.get_special_case(word, tag, stress, ctx)
        if ps is not None:
            return ps, rating
        wl = word.lower()
        if len(word) > 1 and word.replace("'", "").isalpha() and word != word.lower() and (
            tag != "NNP" or len(word) > 7
        ) and word not in self.golds and word not in self.silvers and (
            word == word.upper() or word[1:] == word[1:].lower()
        ) and (
            wl in self.golds or wl in self.silvers or any(
                fn(wl, tag, stress, ctx)[0] for fn in (self.stem_s, self.stem_ed, self.stem_ing)
            )
        ):
            word = wl
        if self.is_known(word, tag):
            return self.lookup(word, tag, stress, ctx)
        elif word.endswith("s'") and self.is_known(word[:-2] + "'s", tag):
            return self.lookup(word[:-2] + "'s", tag, stress, ctx)
        elif word.endswith("'") and self.is_known(word[:-1], tag):
            return self.lookup(word[:-1], tag, stress, ctx)
        _s, rating = self.stem_s(word, tag, stress, ctx)
        if _s is not None:
            return _s, rating
        _ed, rating = self.stem_ed(word, tag, stress, ctx)
        if _ed is not None:
            return _ed, rating
        _ing, rating = self.stem_ing(word, tag, 0.5 if stress is None else stress, ctx)
        if _ing is not None:
            return _ing, rating
        return None, None

    @staticmethod
    def is_currency(word):
        if "." not in word:
            return True
        elif word.count(".") > 1:
            return False
        cents = word.split(".")[1]
        return len(cents) < 3 or set(cents) == {0}

    def get_number(self, word, currency, is_head):
        suffix = re.search(r"[a-z']+$", word)
        suffix = suffix.group() if suffix else None
        word = word[:-len(suffix)] if suffix else word
        result = []
        if word.startswith("-"):
            result.append(self.lookup("minus", None, None, None))
            word = word[1:]

        def extend_num(num, first=True, escape=False):
            for w in re.split(r"[^a-z]+", num if escape else N.cardinal(int(num))):
                if w != "and":
                    result.append(self.lookup(w, None, -2 if w == "point" else None, None))

        if is_digit(word) and suffix in ORDINALS:
            extend_num(N.ordinal(int(word)), escape=True)
        elif not result and len(word) == 4 and currency not in CURRENCIES and is_digit(word):
            extend_num(N.year(int(word)), escape=True)
        elif not is_head and "." not in word:
            num = word.replace(",", "")
            if num[0] == "0" or len(num) > 3:
                for n in num:
                    extend_num(n, first=False)
            elif len(num) == 3 and not num.endswith("00"):
                extend_num(num[0])
                if num[1] == "0":
                    result.append(self.lookup("O", None, -2, None))
                    extend_num(num[2], first=False)
                else:
                    extend_num(num[1:], first=False)
            else:
                extend_num(num)
        elif word.count(".") > 1 or not is_head:
            first = True
            for num in word.replace(",", "").split("."):
                if not num:
                    pass
                elif num[0] == "0" or (len(num) != 2 and any(n != "0" for n in num[1:])):
                    for n in num:
                        extend_num(n, first=False)
                else:
                    extend_num(num, first=first)
                first = False
        elif currency in CURRENCIES and Lexicon.is_currency(word):
            pairs = [(int(num) if num else 0, unit) for num, unit in zip(word.replace(",", "").split("."), CURRENCIES[currency])]
            if len(pairs) > 1:
                if pairs[1][0] == 0:
                    pairs = pairs[:1]
                elif pairs[0][0] == 0:
                    pairs = pairs[1:]
            for i, (num, unit) in enumerate(pairs):
                if i > 0:
                    result.append(self.lookup("and", None, None, None))
                extend_num(num, first=i == 0)
                result.append(self.stem_s(unit + "s", None, None, None) if abs(num) != 1 and unit != "pence" else self.lookup(unit, None, None, None))
        else:
            if is_digit(word):
                word = N.cardinal(int(word))
            elif "." not in word:
                n = int(word.replace(",", ""))
                word = N.ordinal(n) if suffix in ORDINALS else N.cardinal(n)
            else:
                word = word.replace(",", "")
                if word[0] == ".":
                    word = "point " + " ".join(N.cardinal(int(n)) for n in word[1:])
                else:
                    word = N.decimal(word)
            extend_num(word, escape=True)
        if not result:
            return None, None
        result, rating = " ".join(p for p, _ in result), min(r for _, r in result)
        if suffix in ("s", "'s"):
            return self._s(result), rating
        elif suffix in ("ed", "'d"):
            return self._ed(result), rating
        elif suffix == "ing":
            return self._ing(result), rating
        return result, rating

    def append_currency(self, ps, currency):
        if not currency:
            return ps
        currency = CURRENCIES.get(currency)
        currency = self.stem_s(currency[0] + "s", None, None, None)[0] if currency else None
        return f"{ps} {currency}" if currency else ps

    @staticmethod
    def numeric_if_needed(c):
        if not c.isdigit():
            return c
        n = unicodedata.numeric(c)
        return str(int(n)) if n == int(n) else c

    @staticmethod
    def is_number(word, is_head):
        if all(not is_digit(c) for c in word):
            return False
        suffixes = ("ing", "'d", "ed", "'s", *ORDINALS, "s")
        for s in suffixes:
            if word.endswith(s):
                word = word[:-len(s)]
                break
        return all(is_digit(c) or c in ",." or (is_head and i == 0 and c == "-") for i, c in enumerate(word))

    def __call__(self, tk, ctx):
        word = (tk.text if tk.alias is None else tk.alias).replace(chr(8216), "'").replace(chr(8217), "'")
        word = unicodedata.normalize("NFKC", word)
        word = "".join(Lexicon.numeric_if_needed(c) for c in word)
        stress = None if word == word.lower() else self.cap_stresses[int(word == word.upper())]
        ps, rating = self.get_word(word, tk.tag, stress, ctx)
        if ps is not None:
            return apply_stress(self.append_currency(ps, tk.currency), tk.stress), rating
        elif Lexicon.is_number(word, tk.is_head):
            ps, rating = self.get_number(word, tk.currency, tk.is_head)
            return apply_stress(ps, tk.stress), rating
        return None, None
