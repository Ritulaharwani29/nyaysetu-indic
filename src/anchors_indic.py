"""Anchor extraction from Hindi / Marathi / Gujarati text, plus the invariant form of English anchors.

Both sides are reduced to a language-independent vocabulary:
    ACT:IPC  ACT:CRPC  ACT:CPC  ACT:CONST        codes and the Constitution
    ACT:Y1922                                    any other named act -> keyed by its YEAR
    SEC:IPC:302  SEC:CONST:21  SEC:Y1922:16      section / article numbers
    SEC:?:342                                    'weak' anchor: number known, act unknown
                                                 (Marathi/Gujarati use one word for Section and Article)
Act NAMES are not matched across languages (they are translated or transliterated differently);
years, section numbers and the big codes are what survive translation."""
import re

DIG = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")
DEV, GUJ = r"\u0900-\u097F", r"\u0A80-\u0AFF"
W = rf"[{DEV}{GUJ}A-Za-z]*"            # rest of an inflected word (\w does NOT cover Indic vowel signs)

# letter suffixes of section numbers, written in Indic script: 498क -> 498A, 13 बी -> 13B, 80-आई ए -> 80IA
LMAP = {"ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H", "आई": "I",
        "एल": "L", "एम": "M", "एन": "N", "पी": "P", "आर": "R", "एस": "S", "टी": "T", "यू": "U", "व्ही": "V",
        "एक्स": "X", "वाय": "Y", "झेड": "Z", "क": "A", "ख": "B", "ग": "C", "घ": "D", "अ": "A", "ब": "B",
        "એ": "A", "બી": "B", "સી": "C", "ડી": "D", "ઈ": "E", "એફ": "F", "જી": "G", "એચ": "H", "આઈ": "I",
        "એલ": "L", "એમ": "M", "એન": "N", "પી": "P", "આર": "R", "એસ": "S", "ટી": "T", "યુ": "U", "વી": "V",
        "એક્સ": "X", "વાય": "Y", "ઝેડ": "Z", "ક": "A", "ખ": "B", "ગ": "C", "ઘ": "D"}
LET_SORTED = sorted(LMAP, key=len, reverse=True)
LET = "|".join(LET_SORTED)
SUFRX = rf"(?:-?[A-Za-z]{{1,2}}|-?\s?(?:{LET})(?:\s?(?:{LET}))?)"
NOTLETTER = rf"(?![A-Za-z{DEV}{GUJ}])"
# a suffix must not be the start of a dotted abbreviation: "324 आई. पी. सी." (IPC) is NOT 324I, "439 सी. आर. पी. सी." is NOT 439C
NOABBR = rf"(?!\.\s?(?:{LET}|[A-Za-z])(?:\.|\s|$))"


def _suffix(s):
    s = s.replace("-", "").replace(" ", "")
    if re.fullmatch(r"[A-Za-z]+", s):
        return s.upper()
    out, i = "", 0
    while i < len(s):
        for tok in LET_SORTED:
            if s.startswith(tok, i):
                out += LMAP[tok]; i += len(tok); break
        else:
            i += 1
    return out


def split_sentences(text):
    """Split on '।' and on '.!?' + space, but not after short abbreviations ('आय. पी. सी.', 'S. 7')."""
    out, start = [], 0
    for m in re.finditer(r"[।.!?]+\s+", text):
        word = re.search(r"(\S+)$", text[start:m.start()])
        w = re.sub(r"[.\d()\-]", "", word.group(1)) if word else ""
        if text[m.start()] in "।!?" or len(w) >= 3:
            out.append(text[start:m.end()].strip()); start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


CODES = [
    ("IPC", rf"भारतीय\s+दंड\s*(?:संहित|विधान){W}|ભારતીય\s+દંડ\s+સંહિત{W}|"
            rf"आ[यई]\.?\s?पी\.?\s?सी\.?|આ[ઇઈ]\.?\s?પી\.?\s?સી\.?|(?<![A-Za-z])I\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CRPC", rf"(?:फौजदारी|दंड)\s+प्रक्रिया\s+संहित{W}|ફોજદારી\s+(?:કાર્યવાહી|પ્રક્રિયા)\s+સંહિત{W}|"
             rf"सीआरपीसी|सी\.?\s?आर\.?\s?पी\.?\s?सी\.?|સીઆરપીસી|(?<![A-Za-z])Cr\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CPC", rf"(?:सिविल|दिवाणी|दीवानी)\s+प्रक्रिया\s+संहित{W}|(?:સિવિલ|દીવાની)\s+(?:પ્રક્રિયા|કાર્યવાહી)\s+સંહિત{W}|"
            rf"सीपीसी|સીપીસી|(?<![A-Za-z])C\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CONST", rf"संविधान{W}|राज्यघटन{W}|બંધારણ{W}"),
]
CODE_RX = [(c, re.compile(r)) for c, r in CODES]
CONST_INDIA = re.compile(rf"(?:भारत{W}|भारतीय)\s+(?:का\s+|की\s+|चे\s+|च्या\s+)?(?:संविधान|राज्यघटन){W}|ભારત{W}\s+બંધારણ{W}")

ACTWORD = re.compile(rf"अधिनियम{W}|कायद{W}|कायद्य{W}|[अएऍॲ]ॅ?क्ट|[એઍ]ક્ટ|અધિનિયમ{W}|કાયદ{W}")
YEAR_AFTER = re.compile(r"[\s,]*(?:\([^)]{0,40}\)\s*)?(?:ऑफ|ओफ|ઓફ|of)?\s*(1[89]\d\d|20\d\d)(?!\d)")
YEAR_BEFORE = re.compile(rf"(?<!\d)(1[89]\d\d|20\d\d)\s*(?:का|की|के|चा|ची|चे|च्या|ના|ની|નો|નું|નાં)(?![{DEV}{GUJ}])")

MARK = rf"(?:धारा{W}|कलम{W}|કલમ{W}|अनुच्छेद{W}|[Ss]ections?|[Aa]rticles?|(?<![A-Za-z])[Ss]s?\.)"
ONE = rf"\d+(?:{SUFRX}{NOTLETTER}{NOABBR})?(?:\s*\([^)]{{0,12}}\))*"
SEP = r"\s*(?:,|और|तथा|एवं|व|या|अथवा|आणि|किंवा|ते|से|અને|તથા|અથવા|કે|થી|and|or|to|&)\s*"
SECSPAN = re.compile(rf"({MARK})\s*({ONE}(?:{SEP}(?:{MARK}\s*)?{ONE})*)")


def _nums(span):
    span = re.sub(r"\([^)]*\)", " ", span)
    out = []
    for n, suf in re.findall(rf"(\d+)(?:({SUFRX}){NOTLETTER}{NOABBR})?", span):
        if not suf and re.fullmatch(r"\d{4}", n) and 1800 <= int(n) <= 2100:
            continue                               # a year, not a section
        out.append(n + (_suffix(suf) if suf else ""))
    return out


def extract_anchors_indic(text):
    text = (text or "").translate(DIG)
    acts, secs, pending = set(), set(), []
    for sent in split_sentences(text):
        mentions = []                              # (start, end, canonical id)
        for c, rx in CODE_RX:
            mentions += [(m.start(), m.end(), c) for m in rx.finditer(sent)]
        if CONST_INDIA.search(sent):
            acts.add("ACT:CONST")
        ysent = sent                               # a year right after a code name ("संहिता, 1973") belongs to the code,
        for _, e, _c in mentions:                  # not to a later generic word for "Act"/"law"
            ym = re.match(r"[\s,]*(1[89]\d\d|20\d\d)", ysent[e:])
            if ym:
                a0 = e + ym.start(1)
                ysent = ysent[:a0] + "####" + ysent[a0 + 4:]
        for m in ACTWORD.finditer(sent):
            a = YEAR_AFTER.match(ysent, m.end())
            year = a.group(1) if a else None
            if not year:
                ys = YEAR_BEFORE.findall(ysent[max(0, m.start() - 70):m.start()])
                year = ys[-1] if ys else None
            if year:
                mentions.append((m.start(), m.end(), f"Y{year}"))
        for m in SECSPAN.finditer(sent):
            nums = _nums(m.group(2))
            if not nums:
                continue
            if m.group(1).startswith("अनुच्छेद") or m.group(1).lower().startswith("article"):
                secs.update(f"SEC:CONST:{n}" for n in nums); acts.add("ACT:CONST"); continue
            best, bd = None, 10**9                 # nearest act mention; ties -> the preceding one
            for a, b, c in mentions:
                d = a - m.end() if a >= m.end() else (m.start() - b if b <= m.start() else 0)
                d = d + (0 if b <= m.start() else 1)
                if d < bd:
                    best, bd = c, d
            if best:
                secs.update(f"SEC:{best}:{n}" for n in nums)
                if best == "CONST":
                    acts.add("ACT:CONST")
            else:
                pending.append(nums)
        acts.update(f"ACT:{c}" for _, _, c in mentions if c != "CONST")
    weak = set()
    if len(acts) == 1 and pending:                 # orphan sections: attach only if the case names one act
        a = next(iter(acts))[4:]
        for nums in pending:
            secs.update(f"SEC:{a}:{n}" for n in nums)
    else:
        for nums in pending:                       # otherwise keep them as weak anchors (number only)
            weak.update(f"SEC:?:{n}" for n in nums)
    return {"acts": sorted(acts), "secs": sorted(secs), "weak": sorted(weak),
            "all": sorted(acts | secs), "n_unattributed": len(pending) if len(acts) != 1 else 0}


_Y = re.compile(r"^(ACT|SEC):(?:.*_)?ACT_(\d{4})(?::(.+))?$")      # also matches the bare slug ACT_2013


def to_invariant(anchor):
    """English anchor -> the language-independent form (named acts keyed by year)."""
    m = _Y.match(anchor)
    if not m:
        return anchor
    kind, year, sec = m.groups()
    return f"ACT:Y{year}" if kind == "ACT" else f"SEC:Y{year}:{sec}"


def invariant_set(anchor_list):
    return sorted({to_invariant(a) for a in anchor_list})
