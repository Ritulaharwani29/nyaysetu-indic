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
SENT_I = re.compile(r"(?<=[।.!?])\s+")
# clause/suffix letters glued to a section number: 498क -> 498A (Hindi/Gujarati/Marathi lettering)
LETMAP = {"ए": "A", "बी": "B", "सी": "C", "डी": "D", "क": "A", "ख": "B", "ग": "C", "घ": "D", "अ": "A", "ब": "B",
          "એ": "A", "બી": "B", "સી": "C", "ડી": "D", "ક": "A", "ખ": "B", "ગ": "C", "ઘ": "D"}
SUF = r"(?:[A-Za-z]{1,2}|बी|सी|डी|બી|સી|ડી|ए|क|ख|ग|घ|अ|ब|એ|ક|ખ|ગ|ઘ)"

CODES = [
    ("IPC", rf"भारतीय\s+दंड\s*(?:संहित|विधान){W}|ભારતીય\s+દંડ\s+સંહિત{W}|"
            rf"आईपीसी|आई\.?\s?पी\.?\s?सी\.?|આઈપીસી|(?<![A-Za-z])I\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CRPC", rf"(?:फौजदारी|दंड)\s+प्रक्रिया\s+संहित{W}|ફોજદારી\s+(?:કાર્યવાહી|પ્રક્રિયા)\s+સંહિત{W}|"
             rf"सीआरपीसी|सी\.?\s?आर\.?\s?पी\.?\s?सी\.?|સીઆરપીસી|(?<![A-Za-z])Cr\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CPC", rf"(?:सिविल|दिवाणी|दीवानी)\s+प्रक्रिया\s+संहित{W}|(?:સિવિલ|દીવાની)\s+(?:પ્રક્રિયા|કાર્યવાહી)\s+સંહિત{W}|"
            rf"सीपीसी|સીપીસી|(?<![A-Za-z])C\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CONST", rf"संविधान{W}|राज्यघटन{W}|બંધારણ{W}"),
]
CODE_RX = [(c, re.compile(r)) for c, r in CODES]
# "Constitution of India" -> ACT:CONST even when no article/section number is attached
CONST_INDIA = re.compile(rf"(?:भारत{W}|भारतीय)\s+(?:का\s+|की\s+|चे\s+|च्या\s+)?(?:संविधान|राज्यघटन){W}|ભારત{W}\s+બંધારણ{W}")

# keyword that marks "some named Act"; its YEAR is the anchor
ACTWORD = re.compile(rf"अधिनियम{W}|कायद{W}|कायद्य{W}|[अएऍॲ]ॅ?क्ट|[એઍ]ક્ટ|અધિનિયમ{W}|કાયદ{W}")
YEAR_AFTER = re.compile(r"[\s,]*(?:\([^)]{0,40}\)\s*)?(?:ऑफ|ओफ|ઓફ|of)?\s*(1[89]\d\d|20\d\d)(?!\d)")
# year BEFORE the keyword only counts when followed by a genitive particle: "1952 का ... अधिनियम"
YEAR_BEFORE = re.compile(rf"(?<!\d)(1[89]\d\d|20\d\d)\s*(?:का|की|के|चा|ची|चे|च्या|ના|ની|નો|નું|નાં)(?![{DEV}{GUJ}])")

MARK = rf"(?:धारा{W}|कलम{W}|કલમ{W}|अनुच्छेद{W}|[Ss]ections?|[Aa]rticles?|(?<![A-Za-z])ss?\.)"
ONE = rf"\d+(?:-?{SUF}(?![A-Za-z{DEV}{GUJ}]))?(?:\s*\([^)]{{0,12}}\))*"
SEP = r"\s*(?:,|और|तथा|एवं|व|या|आणि|किंवा|ते|से|અને|તથા|અથવા|થી|and|or|to|&)\s*"
SECSPAN = re.compile(rf"({MARK})\s*({ONE}(?:{SEP}(?:{MARK}\s*)?{ONE})*)")


def _nums(span):
    span = re.sub(r"\([^)]*\)", " ", span)
    out = []
    for n, suf in re.findall(rf"(\d+)(?:-?({SUF}))?(?![A-Za-z{DEV}{GUJ}])", span):
        if re.fullmatch(r"\d{4}", n) and 1800 <= int(n) <= 2100:
            continue                               # a year, not a section
        out.append(n + LETMAP.get(suf, suf.upper()))
    return out


def extract_anchors_indic(text):
    text = (text or "").translate(DIG)
    acts, secs, pending = set(), set(), []
    for sent in SENT_I.split(text):
        mentions = []                              # (start, end, canonical id)
        for c, rx in CODE_RX:
            mentions += [(m.start(), m.end(), c) for m in rx.finditer(sent)]
        if CONST_INDIA.search(sent):
            acts.add("ACT:CONST")
        for m in ACTWORD.finditer(sent):
            a = YEAR_AFTER.match(sent, m.end())
            year = a.group(1) if a else None
            if not year:
                ys = YEAR_BEFORE.findall(sent[max(0, m.start() - 70):m.start()])
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
    real_acts = {a for a in acts if a != "ACT:CONST"}
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


_Y = re.compile(r"^(ACT|SEC):.*_ACT_(\d{4})(?::(.+))?$")


def to_invariant(anchor):
    """English anchor -> the language-independent form (named acts keyed by year)."""
    m = _Y.match(anchor)
    if not m:
        return anchor
    kind, year, sec = m.groups()
    return f"ACT:Y{year}" if kind == "ACT" else f"SEC:Y{year}:{sec}"


def invariant_set(anchor_list):
    return sorted({to_invariant(a) for a in anchor_list})
