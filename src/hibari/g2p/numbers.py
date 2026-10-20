"""English number words, the subset of num2words that misaki's number reading uses:
cardinal, ordinal, year and decimal. Word lists only; the caller splits on non-letters."""

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
          "septillion", "octillion", "nonillion", "decillion"]
ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}


def _below_1000(n):
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
        if n:
            words.append("and")
    if n >= 20:
        words.append(TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else ""))
    elif n or not words:
        words.append(ONES[n])
    return words


def cardinal(n):
    """num2words(n): 'one thousand, two hundred and thirty-four'."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 1000:
        return " ".join(_below_1000(n))
    groups = []
    scale = 0
    while n:
        n, g = divmod(n, 1000)
        if g:
            groups.append(" ".join(_below_1000(g)) + (" " + SCALES[scale] if scale else ""))
        scale += 1
    groups.reverse()
    last = groups[-1]
    if len(groups) > 1 and " " not in last.split(" and ")[-1] and "hundred" not in last:
        return ", ".join(groups[:-1]) + " and " + last            # 'one thousand and one'
    return ", ".join(groups)


def ordinal(n):
    """num2words(n, to='ordinal'): 'twenty-third', 'one hundred and first', 'thousandth'."""
    c = cardinal(n)
    head, sep, last = c.rpartition(" ")
    if "-" in last:
        a, _, b = last.rpartition("-")
        last = a + "-" + _ord_word(b)
    else:
        last = _ord_word(last)
    return head + sep + last


def _ord_word(w):
    if w in ORD_IRREGULAR:
        return ORD_IRREGULAR[w]
    if w.endswith("y"):
        return w[:-1] + "ieth"
    return w + "th"


def year(n):
    """num2words(n, to='year'): 'nineteen oh-five', 'two thousand and five', 'twenty ten'."""
    high, low = divmod(n, 100)
    if high == 0 or (high % 10 == 0 and low < 10) or high >= 100:
        return cardinal(n)
    if low == 0:
        lowtext = "hundred"
    elif low < 10:
        lowtext = "oh-" + cardinal(low)
    else:
        lowtext = cardinal(low)
    return cardinal(high) + " " + lowtext


def decimal(s):
    """num2words(float(s)): '3.14' -> 'three point one four'; trailing zeros go like float() does."""
    ip, _, fp = s.partition(".")
    fp = fp.rstrip("0")
    words = cardinal(int(ip or 0))
    if fp:
        words += " point " + " ".join(ONES[int(c)] for c in fp)
    return words
