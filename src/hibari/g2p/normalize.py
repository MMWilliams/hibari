"""Text normalization before tokenizing: readings the lexicon rules can't produce on their own
(times, dates, units, abbreviations, other currencies, symbols, letter/digit mixes). Plain numbers
(cardinals, ordinals, years, decimals, $/£/€ amounts, percents) are read by the lexicon layer, as in
misaki. Word tables live in assets/normalize.json so a native port can reuse them."""
import json, os, re

from . import numbers as N

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
with open(os.path.join(ASSETS, "normalize.json"), encoding="utf-8") as f:
    _T = json.load(f)
MONTHS = _T["months"]                       # "jan" -> "January"
MONTH_NAMES = list(MONTHS.values())
ABBREV = _T["abbreviations"]                # "approx." -> "approximately"
TITLES = _T["titles"]                       # expanded only before a capitalized word: "St Paul"
UNITS = _T["units"]                         # "km" -> ["kilometer", "kilometers"]
SYMBOLS = _T["symbols"]                     # "=" -> "equals"
CURRENCY_WORDS = _T["currency"]             # "¥" -> ["yen", "yen"]
ROMAN_CARDINAL = _T["roman_cardinal"]       # "Chapter IV" -> "Chapter 4"
ROMAN_ORDINAL = _T["roman_ordinal"]         # "Henry VIII" -> "Henry the eighth"


def _alt(words):
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_months = _alt(MONTH_NAMES + [m.capitalize() for m in MONTHS] + ["Sept"])
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
RE_WS = re.compile(r"\s+")
RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s?([AaPp])\.?[Mm]\.?(?![A-Za-z]))?(?!\d)")
RE_DATE_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
RE_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
RE_MONTH_DAY = re.compile(rf"\b({_months})\.? (\d{{1,2}})(?:st|nd|rd|th)?\b(?![\d:%])(,? \d{{4}}\b)?")
RE_DAY_MONTH = re.compile(rf"\b(the )?(\d{{1,2}}) (?:of )?({_months})\b\.?")
RE_UNIT = re.compile(rf"(?<![\w.])(-?{NUM})\s?({_alt(UNITS)})(?![\w])")
RE_DEG = re.compile(rf"(?<![\w.])(-?{NUM})\s?°\s?([CF])?(?![\w])")
RE_CENTS = re.compile(rf"(?<![\w.])({NUM})\s?¢")
RE_CURRENCY = re.compile(rf"(?<![\w])([¥₹₩]|Rs\.?|USD|EUR|GBP)\s?({NUM})(?:\s?(hundred|thousand|million|billion|trillion|[kKmMbB]\b))?")
RE_BIG_MONEY = re.compile(rf"([$£€])({NUM})\s?(hundred|thousand|million|billion|trillion|[kKmMbB]\b)")
RE_TITLE = re.compile(rf"\b({_alt(TITLES)})\.? (?=[A-Z])")
RE_ABBREV = re.compile(rf"(?<![\w.])({_alt(ABBREV)})(?![\w])")
RE_NO = re.compile(r"\b(No|Nos)\. ?(?=\d)")
RE_HASH = re.compile(r"(?<![\w&])#(?=\d)")
ROMAN_NUMERAL = r"((?=[IVXLC])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))"
RE_ROMAN_C = re.compile(rf"\b({_alt(ROMAN_CARDINAL)}) {ROMAN_NUMERAL}\b(?![.'’]\w)")
RE_ROMAN_O = re.compile(rf"\b({_alt(ROMAN_ORDINAL)}) {ROMAN_NUMERAL}\b(?![.'’]\w)")
RE_PHONE = re.compile(r"(?<![\w.\-])(?:\d{3}-)?\d{3}-\d{4}(?![\w.\-])")
RE_RANGE = re.compile(r"(?<![\w.\-])(\d{1,4})\s?[-–]\s?(\d{1,4})(?![\w.\-%])")
RE_FRACTION = re.compile(r"(?<![\w./])(\d{1,2})/(\d{1,2})(?![\w./])")
RE_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
KEEP_SUFFIX = re.compile(r"(?:st|nd|rd|th|s|am|pm|AM|PM|k|K|x|m|M|b|B)\b")
RE_PLUS = re.compile(r"(?<![\w+])\+(?=\d)")
RE_SPACE_PUNCT = re.compile(r" +([,.;:!?])")
FRACTIONS = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _plural(n_text, words):
    return words[0] if n_text.replace(",", "") in ("1", "1.0", "-1") else words[1]


def _roman(s):
    total = 0
    for i, c in enumerate(s):
        v = ROMAN[c]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def _time(m):
    h, mi, sec, ap = m.groups()
    h, mi = int(h), int(mi)
    if h > 24 or mi > 59 or (sec and int(sec) > 59):
        return m.group(0)
    words = N.cardinal(h)
    if mi == 0:
        words += "" if ap or sec else " o'clock"
    elif mi < 10:
        words += " oh " + N.cardinal(mi)
    else:
        words += " " + N.cardinal(mi)
    if sec:
        words += " and " + N.cardinal(int(sec)) + (" second" if int(sec) == 1 else " seconds")
    if ap:                                      # dotted so the lexicon spells the letters (misaki get_NNP)
        words += " " + ap.upper() + ".M" + ("" if m.string[m.end():m.end() + 1] == "." else ".")
    return words


def _date(y, mo, d):
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{MONTH_NAMES[mo - 1]} {N.ordinal(d)}" + (f", {y}" if y else "")


def _date_us(m):
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if len(y) == 2:
        y = ("19" if int(y) > 30 else "20") + y
    mo, d = (a, b) if a <= 12 else (b, a)
    return _date(y, mo, d) or m.group(0)


def _date_iso(m):
    return _date(m.group(1), int(m.group(2)), int(m.group(3))) or m.group(0)


def _month_day(m):
    month, day, year = m.group(1), int(m.group(2)), m.group(3)
    if not 1 <= day <= 31:
        return m.group(0)
    return f"{MONTHS[month.lower()[:3]]} {N.ordinal(day)}" + (year or "")


def _day_month(m):
    day, month = int(m.group(2)), m.group(3)
    if not 1 <= day <= 31:
        return m.group(0)
    return f"the {N.ordinal(day)} of {MONTHS[month.lower()[:3]]}"


def _unit(m):
    n, u = m.group(1), m.group(2)
    return f"{n} {_plural(n, UNITS[u])}"


def _deg(m):
    n, scale = m.group(1), m.group(2)
    words = f"{n} {_plural(n, ['degree', 'degrees'])}"
    return words + ({"C": " Celsius", "F": " Fahrenheit"}[scale] if scale else "")


def _currency(m):
    sym, n, big = m.groups()
    words = CURRENCY_WORDS[sym.rstrip(".")]
    big = {"k": "thousand", "m": "million", "b": "billion"}.get((big or "").lower(), big)
    return f"{n} {big} {words[1]}" if big else f"{n} {_plural(n, words)}"


def _big_money(m):
    sym, n, big = m.groups()
    big = {"k": "thousand", "m": "million", "b": "billion"}.get(big.lower(), big)
    return f"{n} {big} {CURRENCY_WORDS[sym][1]}"


def _fraction(m):
    a, b = int(m.group(1)), int(m.group(2))
    if 0 < a < b <= 10:
        unit = FRACTIONS[b]
        if unit == "half":
            return "one half" if a == 1 else f"{a} halves"
        return f"{a} {unit}" + ("" if a == 1 else "s")
    if 1 <= a <= 12 and 1 <= b <= 31:
        return _date(None, a, b)
    return f"{a} slash {b}"


def normalize(text):
    text = RE_WS.sub(" ", text).strip()
    if not text:
        return text
    text = text.replace(" ", " ").replace("’", "'").replace("‒", "-").replace("―", "—")
    text = RE_TIME.sub(_time, text)
    text = RE_DATE_ISO.sub(_date_iso, text)
    text = RE_DATE_US.sub(_date_us, text)
    text = RE_BIG_MONEY.sub(_big_money, text)
    text = RE_CENTS.sub(lambda m: f"{m.group(1)} {_plural(m.group(1), CURRENCY_WORDS['¢'])}", text)
    text = RE_CURRENCY.sub(_currency, text)
    text = RE_DEG.sub(_deg, text)
    text = RE_UNIT.sub(_unit, text)
    text = RE_MONTH_DAY.sub(_month_day, text)
    text = RE_DAY_MONTH.sub(_day_month, text)
    text = RE_TITLE.sub(lambda m: TITLES[m.group(1)] + " ", text)
    text = RE_ABBREV.sub(lambda m: ABBREV[m.group(1)] + ("." if m.end() == len(m.string) and m.group(1).endswith(".") else ""), text)
    text = RE_NO.sub(lambda m: "Number " if m.group(1) == "No" else "Numbers ", text)
    text = RE_HASH.sub("number ", text)
    text = RE_ROMAN_C.sub(lambda m: f"{m.group(1)} {_roman(m.group(2))}", text)
    text = RE_ROMAN_O.sub(lambda m: f"{m.group(1)} the {N.ordinal(_roman(m.group(2)))}", text)
    text = RE_PHONE.sub(lambda m: " ".join(m.group(0).replace("-", ", ")), text)
    text = RE_FRACTION.sub(_fraction, text)
    text = RE_RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", text)
    text = RE_PLUS.sub("plus ", text)
    for sym, word in SYMBOLS.items():
        if sym in text:
            pat = rf"(?<=\S) ?{re.escape(sym)} ?(?=\S)" if sym in "=<>~" else re.escape(sym)
            text = re.sub(pat, f" {word} " if word else " ", text)
    return RE_SPACE_PUNCT.sub(r"\1", RE_WS.sub(" ", text)).strip()
