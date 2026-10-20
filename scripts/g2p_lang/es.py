"""Spanish grapheme-to-phoneme prototype: letter rules + stress rules + a number reader, written to
reproduce what Kokoro's Spanish voices were trained on (espeak-ng `es` through misaki's EspeakG2P
mapping: Castilian θ, allophones β ð ɣ, ʎ for ll, ʝ for y, diphthongs ei->A ai->I au->W).
No espeak code or data is used; the rules were written from Spanish orthography and tuned on
espeak's output. About 10 KB of code and tables, no lexicon.

    from es import phonemize; phonemize("Tengo 25 años.")  -> 'tˈɛŋɡo βˌAntiθˈinko ˈaɲos.'
"""
import re, unicodedata

VOW = "aeiouáéíóúü"
NUCLEUS = "aeiouAIOWYɪʊɛ"                      # phones that can carry a syllable
P, S = "ˈ", "ˌ"
UNSTRESSED = set("el la los las de del en con a al y e o u que se su sus mi mis tu tus lo le les me te nos os por sin "
                 "si tras ante quien quienes cuyo cuya cuyos cuyas".split())
CONJ = set("y e o u".split())                    # never stressed, even before punctuation
WEAK = set("como para entre sobre hasta pero cuando donde hacia aunque desde mientras porque contra bajo cuanto".split())                     # secondary stress on the first syllable
EXCEPTIONS = {"ser": "sˈer", "méxico": "mˈɛxiko", "muy": "mˈuj", "mexico": "mˈɛxiko", "etc": "etθˈeteɾa"}
UNITS = ["θˈeɾo", "ˈuno", "dˈos", "tɾˈes", "kwˈatɾo", "θˈinko", "sˈAs", "sjˈete", "ˈoʧo", "nwˈeβe", "djˈɛθ", "ˈonθe", "dˈoθe",
         "tɾˈeθe", "katˈoɾθe", "kˈinθe"]
TENS = {2: "bˈAnte", 3: "tɾˈAnta", 4: "kwaɾˈɛnta", 5: "θinkwˈɛnta", 6: "sɛsˈɛnta", 7: "sɛtˈɛnta", 8: "oʧˈɛnta", 9: "noβˈɛnta"}
HUNDREDS = {1: "θjˈento", 2: "dosθjˈentos", 3: "tɾesθjˈentos", 4: "kwˌatɾoθjˈentos", 5: "kinjˈɛntos", 6: "sAsθjˈentos",
            7: "sˌeteθjˈɛntos", 8: "oʧoθjˈentos", 9: "nˌoβeθjˈɛntos"}
LETTERS = {"a": "ˈa", "b": "bˈe", "c": "θˈe", "d": "dˈe", "e": "ˈe", "f": "ˈefe", "g": "xˈe", "h": "ˈaʧe", "i": "ˈi", "j": "xˈota",
           "k": "kˈa", "l": "ˈele", "m": "ˈeme", "n": "ˈene", "o": "ˈo", "p": "pˈe", "q": "kˈu", "r": "ˈere", "s": "ˈese",
           "t": "tˈe", "u": "ˈu", "v": "ˈuβe", "w": "ˈuβe ðˈoβle", "x": "ˈekis", "y": "ʝˈe", "z": "θˈeta"}
PUNCT = '.,;:!?¿¡()"“”«»…'
ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _sec(ps):
    return ps.replace(P, S)


def number(n):
    """Phonemes (one or more words) for an integer, as espeak reads digits: tens and units are joined with
    'i' and the tens keep secondary stress; hundreds, mil, millón are separate words."""
    if n < 16:
        return [UNITS[n]]
    if n < 20:
        return [_sec(UNITS[10]) + "i" + UNITS[n - 10]]
    if n < 30:
        return ["bˌAnti" + UNITS[n - 20]] if n > 20 else [TENS[2]]
    if n < 100:
        t, u = divmod(n, 10)
        return [TENS[t]] if u == 0 else [_sec(TENS[t]) + "i" + UNITS[u]]
    if n < 1000:
        h, r = divmod(n, 100)
        head = ["θjˈen"] if n == 100 else [HUNDREDS[h]]
        return head + (number(r) if r else [])
    if n < 1000000:
        k, r = divmod(n, 1000)
        return (["mˈil"] if k == 1 else number(k) + ["mˈil"]) + (number(r) if r else [])
    m, r = divmod(n, 1000000)
    head = ["ˈunmiʝˈon"] if m == 1 else number(m) + ["miʝˈones"]
    return head + (number(r) if r else [])


def _roman(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        total += -v[c] if i + 1 < len(s) and v[s[i + 1]] > v[c] else v[c]
    return total


def _letters(w):
    """Orthographic word -> segments: consonant phonemes and vowel letters (vowels keep their letter so the
    stress rules can see accents and diphthongs). 'R' = a written single r (tap or trill decided later)."""
    out = []
    i, n = 0, len(w)
    while i < n:
        c, nx = w[i], w[i + 1] if i + 1 < n else ""
        two = c + nx
        if two == "ch": out.append("ʧ"); i += 2
        elif two == "ll": out.append("ʎ"); i += 2
        elif two == "rr": out.append("r"); i += 2
        elif two == "qu":
            out.append("k"); i += 2
            if w[i: i + 1] in ("a", "o", "á", "ó"): out.append("w")
        elif two == "ck": out.append("k"); i += 2
        elif two == "gu" and w[i + 2: i + 3] in ("e", "i", "é", "í"): out.append("ɡ"); i += 2
        elif c == "c": out.append("θ" if nx and nx in "eiéí" else "k"); i += 1
        elif c == "g": out.append("x" if nx and nx in "eiéí" else "ɡ"); i += 1
        elif c == "z": out.append("θ"); i += 1
        elif c == "j": out.append("x"); i += 1
        elif c == "ñ": out.append("ɲ"); i += 1
        elif c == "v": out.append("b"); i += 1
        elif c == "x": out.append("s" if i == 0 else "ks"); i += 1
        elif two == "ps" and i == 0: i += 1                     # psicología: silent p
        elif c == "h": i += 1
        elif c == "y":
            out.append("ʝ" if nx and nx in VOW else "i")       # y, rey, hoy: vowel i (diphthong rules below)
            i += 1
        elif c == "r": out.append("R"); i += 1
        elif c in "bdfklmnpstw": out.append(c); i += 1
        elif c in VOW: out.append(c); i += 1
        else: i += 1
    return out


def _nuclei(seg):
    """Vowel letters -> syllable nuclei and glides. Returns (phones, nucleus indexes, accented nucleus index)."""
    out, nuc, acc = [], [], None
    i = 0
    while i < len(seg):
        s = seg[i]
        if s not in VOW:
            out.append(s); i += 1; continue
        j = i
        while j < len(seg) and seg[j] in VOW:
            j += 1
        run = seg[i:j]
        prev = out[-1] if out else ""
        prev2 = out[-2] if len(out) > 1 else ""
        # after a trill, ʎ, or a consonant cluster (bl, gr, tr...) i/u before a vowel are written as vowels
        # (but still count as a glide for the stress rule); "ui" is a hiatus after l or r
        cluster = (prev == "R" and (prev2 == "" or prev2 not in VOW)) or (prev in "lR" and prev2 and prev2 not in VOW) or prev == "ʎ"
        k = 0
        while k < len(run):
            v = run[k]
            nx = run[k + 1] if k + 1 < len(run) else ""
            base = unicodedata.normalize("NFD", v)[0]
            accented = v in "áéíóú"
            nx_base = unicodedata.normalize("NFD", nx)[0] if nx else ""
            if v in "iu" and nx and nx_base != base and not accented:
                if cluster or (v + nx_base == "ui" and prev in "lR"):
                    out.append(v); nuc.append(len(out) - 1)        # hiatus: a syllable of its own
                else:
                    out.append("j" if v == "i" else "w")           # rising diphthong
                k += 1; prev = out[-1]; continue
            if v == "ü":
                out.append("u"); nuc.append(len(out) - 1); k += 1; prev = "u"; continue
            if base in "aeo" and nx and nx in "iu":              # falling diphthong
                dip = {"ai": "I", "ei": "A", "oi": "oɪ", "au": "W", "eu": "eʊ", "ou": "ow"}[base + nx]
                if accented:
                    acc = len(out)
                out.append(dip); nuc.append(len(out) - 1); k += 2; prev = dip[-1]; continue
            if accented:
                acc = len(out)
            out.append(base); nuc.append(len(out) - 1); k += 1; prev = base
        i = j
    return out, nuc, acc


def _stress_word(w, weak=False):
    ph, nuc, acc = _nuclei(_letters(w))
    if not nuc:
        return ph
    if weak:
        marks = {nuc[0]: S}
    else:
        if acc is not None:
            pi = nuc.index(acc)
        elif len(nuc) == 1:
            pi = 0
        else:
            pi = len(nuc) - 2 if w[-1] in "aeiounsy" else len(nuc) - 1
        marks = {nuc[pi]: P}
        for q in range(0, pi - 1, 2):                         # secondary: 1st, 3rd, ... never adjacent to primary
            marks[nuc[q]] = S
    out = []
    for i, p in enumerate(ph):
        if i in marks:
            out.append(marks[i])
        out.append(p)
    # stressed e before n + consonant -> ɛ (tengo, ciento, puente)
    for i, p in enumerate(out):
        if p == "e" and i > 0 and out[i - 1] == P and out[i + 1: i + 2] == ["n"] and i + 2 < len(out) and out[i + 2] not in NUCLEUS + "jw":
            out[i] = "ɛ"
    # r: trill at word start and after n l s, else a tap
    return _r(out)


def _r(out):
    """r: trill at word start and after n l s, else a tap."""
    for i, p in enumerate(out):
        if p == "R":
            pv = next((q for q in reversed(out[:i]) if q not in (P, S)), "")
            out[i] = "r" if pv == "" or pv in "nls" else "ɾ"
    return out


def word_phones(w, final=False):
    """Phoneme list for one lowercase word, with stress, before the allophone rules. final: the word ends a
    phrase (before punctuation), where function words get their stress back."""
    if w in EXCEPTIONS:
        return list(EXCEPTIONS[w])
    if w in UNSTRESSED and (not final or w in CONJ):
        return _r(_nuclei(_letters(w))[0])
    if w in WEAK:
        return _stress_word(w, weak=True)
    if w.endswith("mente") and len(w) > 7:
        return _stress_word(w[:-5]) + ["m", P, "e", "n", "t", "e"]
    return _stress_word(w)


def _allophones(flat):
    """Sentence-level rules over the flat phone list (' ' between words, '|' at pauses): b d ɡ -> β ð ɣ after
    anything but a nasal or a pause, nasal assimilation (also across words), then the markers are dropped."""
    phones = [p for p in flat]
    out = []
    prev = ""
    for i, p in enumerate(phones):
        nx = next((q for q in phones[i + 1:] if q not in (P, S, " ")), "")
        if p in (P, S, " "):
            out.append(p); continue
        if p == "|":
            prev = ""; continue
        if p == "b" and prev and prev not in "mnŋ" and nx not in ("", "|", "b", "β", "d", "ð", "t", "p", "k"):
            p = "β"
        elif p == "d" and prev and prev not in "mnŋ" and nx and nx[0] in NUCLEUS + "jwɾlβðɣrxʝʎɲ":
            p = "ð"
        elif p == "ɡ" and prev and prev not in "mnŋ" and nx not in ("", "|", "n", "m"):
            p = "ɣ"
        elif p == "p" and nx == "t":
            out.append(p); out.append("ː"); prev = p; continue
        elif p == "n" and (nx in ("b", "β", "p") or (nx == "f" and next((q for q in phones[i + 2:] if q not in (P, S, " ", "f")), "") == "l")):
            p = "m"
        elif p == "n" and nx in ("ɡ", "ɣ", "x"):
            p = "ŋ"
        out.append(p)
        prev = p
    return out


_TOKEN = re.compile(r"\d+|[^\W\d_]+|\s+|.", re.U)


def phonemize(text):
    text = text.replace("«", "“").replace("»", "”")
    flat = []
    toks = [m.group() for m in _TOKEN.finditer(text.strip())]
    for i, t in enumerate(toks):
        if t.isspace():
            flat.append(" "); continue
        if t.isdigit():
            if len(t) > 1 and t[0] == "0":
                words = [w for d in t for w in number(int(d))]
            else:
                words = number(int(t))
            for k, wp in enumerate(words):
                flat += list(wp) + ([" "] if k < len(words) - 1 else [])
            continue
        if t[0].isalpha():
            if len(t) == 1 and t.lower() in LETTERS and t.isupper():
                flat += list(LETTERS[t.lower()]); continue
            if t.isupper() and len(t) > 1 and ROMAN.match(t):
                words = number(_roman(t))
                for k, wp in enumerate(words):
                    flat += list(wp) + ([" "] if k < len(words) - 1 else [])
                continue
            nxt = next((x for x in toks[i + 1:] if not x.isspace()), "")
            flat += word_phones(t.lower(), final=(nxt == "" or nxt in ".,;:!?…")); continue
        if t == "/":
            flat += list(" βˈara "); continue
        if t == "%":
            flat += list(" poɾθjˈɛnto"); continue
        if t == "-":
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            prv = toks[i - 1] if i > 0 else ""
            flat.append(" " if (prv.isdigit() and nxt.isdigit()) else "|" if (prv.isspace() or nxt.isspace()) else "")
            continue
        flat.append(t)
        if t in PUNCT:
            flat.append("|")
    out = "".join(_allophones(flat))
    return re.sub(r" +", " ", out).replace(" .", " .").strip()


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["Tengo 25 años y 3 hijos.", "El 19 de mayo de 1895 Martí cayó herido en la batalla de Dos Ríos."]:
        print(phonemize(t))
