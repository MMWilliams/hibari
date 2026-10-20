"""Word tokenizer with spaCy's English rules (spaCy, MIT): special cases (contractions, 'Mr.',
'10am'), then prefixes and suffixes are peeled off each whitespace-separated chunk, then infixes
(hyphens, slashes) split what is left; finally adjacent pieces that spell a special case are joined
back ('Sept' + '.' -> 'Sept.'). The character classes are written out for Latin text."""
import re

L = r"[^\W\d_]"                       # a letter
LO, UP = "a-zß-öø-ÿ", "A-ZÀ-ÖØ-Þ"
Q = "'\"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉"
PUNCT = "…,:;!?¿¡()[]{}<>_#*&。？！，、；：～·"
ICONS = "™®©°§¶†‡•"
CURRENCY = r"\$|£|€|¥|฿|US\$|C\$|A\$|₽|₹|₩|₪|₺|₴|₦|₫|₱|₲|₵|₸|₼|₾|₿"
UNITS = (r"km|km²|km³|m|m²|m³|dm|dm²|dm³|cm|cm²|cm³|mm|mm²|mm³|ha|µm|nm|yd|in|ft|kg|g|mg|µg|t|lb|oz|m/s|km/h"
         r"|kmh|mph|hPa|Pa|mbar|mb|MB|kb|KB|gb|GB|tb|TB|T|G|M|K|%")
_punct = [re.escape(c) for c in PUNCT]
_quotes = [re.escape(c) for c in Q]
_icons = [f"[{re.escape(ICONS)}]"]
PREFIXES = (["§", "%", "=", "—", "–", r"\+(?![0-9])", "…", "……"] + _punct + [r"\.\.+", "…"] + _quotes
            + [CURRENCY] + _icons)
SUFFIXES = (["…", "……"] + _punct + [r"\.\.+", "…"] + _quotes + _icons + ["'s", "'S", "’s", "’S", "—", "–"] + [
    r"(?<=[0-9])\+", r"(?<=°[FfCcKk])\.", rf"(?<=[0-9])(?:{CURRENCY})", rf"(?<=[0-9])(?:{UNITS})",
    rf"(?<=[0-9{LO}%²\-\+{re.escape(PUNCT)}{re.escape(Q)}])\.", rf"(?<=[{UP}][{UP}])\."])
INFIXES = ([r"\.\.+", "…"] + _icons + [r"(?<=[0-9])[+\-\*^](?=[0-9-])",
           rf"(?<=[{LO}{re.escape(Q)}])\.(?=[{UP}{re.escape(Q)}])",
           rf"(?<={L}),(?={L})", rf"(?<=[^\W_])(?:-|–|—|--|---|——|~)(?={L})", rf"(?<=[^\W_])[:<>=/](?={L})"])
URL = re.compile(r"^(?:(?:https?|ftp)://|www\.)\S+$|^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$"
                 r"|^(?:[\w-]+\.)+(?:com|org|net|edu|gov|io|co|uk|de|fr|info|biz|us|ca|au|tv|me|ly|ai)(?:/\S*)?$", re.I)


class Tokenizer:
    def __init__(self, exceptions):
        self.special = exceptions                                       # text -> [token texts]
        self.prefix = re.compile("|".join(f"^(?:{p})" for p in PREFIXES))
        self.suffix = re.compile("|".join(f"(?:{s})$" for s in SUFFIXES))
        self.infix = re.compile("|".join(f"(?:{i})" for i in INFIXES))

    def __call__(self, text):
        """-> [(token text, whitespace)]; whitespace is ' ' when the token was followed by any."""
        out = []
        for m in re.finditer(r"\S+", text):
            toks = self.split(m.group())
            ws = " " if m.end() < len(text) else ""
            out += [(t, "") for t in toks[:-1]] + [(toks[-1], ws)]
        return out

    def split(self, s):
        special = self.special
        if s in special:
            return list(special[s])
        pre, suf = [], []
        last = -1
        while s and len(s) != last and s not in special:
            last = len(s)
            m = self.prefix.search(s)
            pre_len = m.end() if m else 0
            if pre_len:
                prefix, minus_pre = s[:pre_len], s[pre_len:]
                if minus_pre and minus_pre in special:
                    s = minus_pre
                    pre.append(prefix)
                    break
            m = self.suffix.search(s[pre_len:])
            suf_len = m.end() - m.start() if m else 0
            if suf_len:
                suffix, minus_suf = s[-suf_len:], s[:-suf_len]
                if minus_suf and minus_suf in special:
                    s = minus_suf
                    suf.append(suffix)
                    break
            if pre_len and suf_len and pre_len + suf_len <= len(s):
                s = s[pre_len:-suf_len]
                pre.append(prefix)
                suf.append(suffix)
            elif pre_len:
                s = minus_pre
                pre.append(prefix)
            elif suf_len:
                s = minus_suf
                suf.append(suffix)
        toks = pre
        if s:
            if s in special:
                toks += special[s]
            elif URL.match(s):
                toks.append(s)
            else:
                start = 0
                for m in self.infix.finditer(s):
                    if m.start() == 0 or m.start() == m.end():
                        continue
                    if m.start() > start:
                        toks.append(s[start:m.start()])
                    toks.append(m.group())
                    start = m.end()
                if start < len(s):
                    toks.append(s[start:])
        toks += reversed(suf)
        return self.rejoin(toks) if len(toks) > 1 else toks

    def rejoin(self, toks):
        """spaCy's special-case pass over the split pieces: 'Sept' '.' -> 'Sept.', ')' ':' -> '):'."""
        special = self.special
        out, i = [], 0
        while i < len(toks):
            for k in (3, 2):
                if i + k <= len(toks):
                    joined = "".join(toks[i:i + k])
                    if joined in special:
                        out += special[joined]
                        i += k
                        break
            else:
                out.append(toks[i])
                i += 1
        return out
