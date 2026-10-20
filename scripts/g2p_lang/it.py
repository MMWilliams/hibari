"""Italian grapheme-to-phoneme prototype (rules only, no lexicon), written to approximate what Kokoro's
Italian voices were trained on (espeak-ng `it` through misaki's EspeakG2P mapping). Italian spelling
leaves three things to the lexicon: stress position (default penultimate here), open vs closed e/o
(ɛ/e, ɔ/o: stressed e/o are left closed here) and voiced vs voiceless s/z (s between vowels -> z,
z -> ʦ, as espeak mostly does). Those three are where a lexicon-free prototype loses.

    from it import phonemize; phonemize("Ho 25 anni.")
"""
import re, unicodedata

VOW = "aeiouàèéìòóù"
P, S = "ˈ", "ˌ"
WEAK = set("della delle dello degli una come questo questa questi queste quella quelle quello quelli sono al allo alla alle "
           "ai agli sul sullo sulla sulle sui dal dallo dalla dalle dai dagli anche anno era loro essere possono furono ogni "
           "verso".split())
PRIMARY_WEAK = set("nello nella nelle negli".split())
EXCEPTIONS = {"i": "i", "un": "ʊn", "uno": "ˈuno", "più": "pjʊ", "fu": "fʊ", "gli": "ʎɪ", "dei": "dˌA", "suo": "sˌʊo", "sua": "sˌʊa",
              "sue": "sˌʊe", "suoi": "sˌʊoi", "può": "pʊˈɔ", "due": "dˈue", "cui": "kˈuj", "poi": "pˈɔi", "sia": "sˌia",
              "re": "rˈɛ", "è": "e"}
NO_STRESS = set("di da de il lo la le i l e ed o a ad in con su per tra fra che chi se non si mi ti ci vi ne li ma del "
                "dell col nel al dal sul ai dai nei".split())
UNITS = ["ʣˈɛɾo", "ˈuno", "dˈue", "trˈe", "kwˈatːro", "ʧˈinkwe", "sˈɛj", "sˈɛtːe", "ˈɔtːo", "nˈɔve", "djˈɛʧɪ", "ˈundiʧɪ", "dˈodiʧɪ",
         "trˈediʧɪ", "kwatːˈordiʧɪ", "kwˈindiʧɪ", "sˈediʧɪ", "diʧaSˈɛtːe", "diʧˈɔtːo", "diʧanːˈɔve"]
TENS = {2: "vˈentɪ", 3: "trˈeːnta", 4: "kʊaɾˈaːnta", 5: "ʧinkwˈanta", 6: "seSˈanta", 7: "setːˈanta", 8: "otːˈanta", 9: "novˈanta"}
HUND = {1: "ʧˈɛnto", 2: "dʊeʧˈɛnto", 3: "treʧˈɛnto", 4: "kʊatːroʧˈɛnto", 5: "ʧɪnkʊeʧˈɛnto", 6: "sejʧˈɛnto", 7: "setːeʧˈɛnto",
        8: "otːoʧˈɛnto", 9: "noveʧˈɛnto"}
PUNCT = '.,;:!?()"“”«»…'


def _sec(ps):
    return ps.replace(P, S)


def _join(a, b):
    """venti + uno -> ventuno, trenta + otto -> trentotto (final vowel dropped before 1 and 8)."""
    a = re.sub(r"[ɪa]$", "", a) if b.lstrip(P) and b.lstrip(P)[0] in "uɔ" else a
    return _sec(a).replace("ː", "") + b


def number(n):
    if n < 20:
        return [UNITS[n]]
    if n < 100:
        t, u = divmod(n, 10)
        return [TENS[t]] if u == 0 else [_join(TENS[t], UNITS[u] if u != 3 else "trˈe")]
    if n < 1000:
        h, r = divmod(n, 100)
        return [HUND[h]] + (number(r) if r else [])
    if n < 1000000:
        k, r = divmod(n, 1000)
        head = ["mˈille"] if k == 1 else number(k) + ["mˈiːla"]
        return head + (number(r) if r else [])
    m, r = divmod(n, 1000000)
    return (["ˈun miliˈone"] if m == 1 else number(m) + ["miliˈonɪ"]) + (number(r) if r else [])


def _letters(w):
    out = []
    i, n = 0, len(w)
    while i < n:
        c, nx = w[i], w[i + 1] if i + 1 < n else ""
        two, three = c + nx, w[i: i + 3]
        if two == "ch": out.append("k"); i += 2
        elif two == "gh": out.append("ɡ"); i += 2
        elif three in ("gli",) and (i + 3 == n or w[i + 3] in VOW): out.append("ʎ"); i += 3
        elif two == "gn": out.append("ɲ"); i += 2
        elif two == "sc" and w[i + 2: i + 3] and w[i + 2] in "eiéèì": out.append("ʃ"); i += 2
        elif two == "cc" and w[i + 2: i + 3] and w[i + 2] in "eiéèì": out.append("ʧ"); out.append("ʧ"); i += 2
        elif two == "gg" and w[i + 2: i + 3] and w[i + 2] in "eiéèì": out.append("ʤ"); out.append("ʤ"); i += 2
        elif c == "c" and nx and nx in "eiéèì":
            out.append("ʧ"); i += 1
            if nx == "i" and w[i + 1: i + 2] and w[i + 1] in "aeouàèéòóù": i += 1      # cia, cio: silent i
        elif c == "g" and nx and nx in "eiéèì":
            out.append("ʤ"); i += 1
            if nx == "i" and w[i + 1: i + 2] and w[i + 1] in "aeouàèéòóù": i += 1
        elif two == "sc" and w[i + 2: i + 3] == "i" and w[i + 3: i + 4] and w[i + 3] in "aeouàèéòóù": out.append("ʃ"); i += 3
        elif c == "c": out.append("k"); i += 1
        elif c == "g": out.append("ɡ"); i += 1
        elif two == "qu": out.append("k"); out.append("w"); i += 2
        elif c == "z": out.append("ʣ" if i == 0 else "ʦ"); i += 1
        elif c == "s":
            prev = out[-1] if out else ""
            if nx and nx in VOW and prev and prev in VOW:
                out.append("z")
            elif nx and nx in "bdgvlmnr":
                out.append("z")
            else:
                out.append("s")
            i += 1
        elif c == "h": i += 1
        elif c == "x": out.append("k"); out.append("s"); i += 1
        elif c == "y": out.append("j" if nx and nx in VOW else "i"); i += 1
        elif c == "j": out.append("j"); i += 1
        elif c in "bdfklmnprtvw": out.append(c); i += 1
        elif c in VOW: out.append(c); i += 1
        else: i += 1
    return out


def _nuclei(seg):
    """Vowels: unstressed i/u before a vowel become glides j/w (except after a cluster, where espeak keeps the
    vowel); the stressed syllable gets its mark; final unstressed i -> ɪ, unstressed u -> ʊ."""
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
        cluster = prev in "lrʦʣ" and (prev in "ʦʣ" or (prev2 and prev2 not in VOW))
        k = 0
        while k < len(run):
            v = run[k]
            nx = run[k + 1] if k + 1 < len(run) else ""
            base = unicodedata.normalize("NFD", v)[0]
            accented = v in "àèéìòóù"
            if accented:
                acc = len(out)
            final = j == len(seg) and k + 2 == len(run)
            if base == "i" and nx and not accented and nx not in "iu" and (cluster or final) and prev and prev not in VOW:
                out.append("i"); k += 1; continue                  # written as a vowel but not a syllable (stˈɔria)
            if base == "i" and nx and not accented and nx not in "iu":
                out.append("j"); k += 1; continue
            if base == "u" and nx and not accented and nx not in "iu" and prev in "kɡ":
                out.append("w"); k += 1; continue
            if base in "aeo" and nx == "u":
                out.append(base); nuc.append(len(out) - 1); out.append("ʊ"); k += 2; continue
            if base in "ao" and nx == "i" and k + 2 >= len(run):
                out.append(base); nuc.append(len(out) - 1); out.append("j"); k += 2; continue
            out.append(base); nuc.append(len(out) - 1); k += 1
        i = j
    return out, nuc, acc


def word_phones(w):
    ph, nuc, acc = _nuclei(_letters(w))
    if not nuc:
        return ph
    if w in EXCEPTIONS:
        return list(EXCEPTIONS[w])
    if w in NO_STRESS:
        marks = {}
    elif w in WEAK:
        marks = {nuc[0]: S}
    elif w in PRIMARY_WEAK:
        marks = {nuc[0]: P}
    else:
        pi = nuc.index(acc) if acc is not None else (0 if len(nuc) == 1 else len(nuc) - 2)
        marks = {nuc[pi]: P}
    stressed = {i for i, m in marks.items() if m == P}
    out = []
    for i, p in enumerate(ph):
        if i in marks:
            out.append(marks[i])
        if p == "i" and i not in stressed and i == len(ph) - 1:
            p = "ɪ"
        elif p == "n" and i + 1 < len(ph) and ph[i + 1] == "ɡ":
            p = "ŋ"
        elif p == "ɲ" and 0 < i < len(ph) - 1 and ph[i - 1] in VOW:
            out.append(p)
        elif p == "ʎ" and i == len(ph) - 1:
            out.append(p); p = "ɪ"
        elif p == "u" and i not in stressed:
            p = "ʊ"
        elif p == "e" and i in stressed and acc is not None and w[-1] == "è":
            p = "ɛ"
        elif p == "o" and i in stressed and acc is not None and w[-1] == "ò":
            p = "ɔ"
        out.append(p)
    # double consonants: stops and affricates get ː, sonorants are written twice (as espeak does)
    res = []
    for p in out:
        if res and p == res[-1] and p not in VOW and p not in (P, S):
            res.append("ː" if p in "ptkbdɡfvʧʤʦʣʃ" else p)
        else:
            res.append(p)
    return res


_TOKEN = re.compile(r"\d+|[^\W\d_]+|'|\s+|.", re.U)


def phonemize(text):
    text = text.replace("«", "“").replace("»", "”")
    out = []
    for m in _TOKEN.finditer(text.strip()):
        t = m.group()
        if t.isspace():
            out.append(" "); continue
        if t.isdigit():
            out.append(" ".join(number(int(t)))); continue
        if t[0].isalpha():
            out.append("".join(word_phones(t.lower()))); continue
        if t == "'":
            continue
        out.append(t)
    return re.sub(r" +", " ", "".join(out)).replace("S", "S").strip().replace("ss", "S")


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["Ho 25 anni e 3 figli.", "Le prime bombe caddero su Isola del Liri il 23 ottobre del 1943."]:
        print(phonemize(t))
