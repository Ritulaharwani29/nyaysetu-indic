import re

STOP = {"the", "of", "and", "for", "to", "in"}

# Abbreviated codes that have no "Act" in their name
ALIASES = [
    ("IPC",  r"indian\s+penal\s+code|(?<![A-Za-z])I\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CRPC", r"code\s+of\s+criminal\s+procedure|(?<![A-Za-z])Cr\.?\s?P\.?\s?C\.?(?![A-Za-z])|(?<![A-Za-z])Cr\.?\s?C\.?(?![A-Za-z])"),
    ("CPC",  r"code\s+of\s+civil\s+procedure|(?<![A-Za-z])C\.?\s?P\.?\s?C\.?(?![A-Za-z])"),
    ("CONST", r"constitution\s+of\s+india"),
]
ALIAS_RX = [(c, re.compile(r, re.I)) for c, r in ALIASES]

# "<words> Act, 1988"  (generic named statutes; lowercase words and "Act. 1987" are allowed)
GENERIC = re.compile(
    r"((?:[A-Za-z(][^\s,;:]*\s+(?:(?:of|and|the|for|to)\s+)?){1,9})Act\s*[,.]?\s*(?:\([IVXLC]+\s+of\s+\d{4}\)\s*)?(?:of\s+)?(\d{4})\b")

# one section number: 302, 498A, 120-B, 376DB; NOT the first letters of a following word ("302IPC" -> 302)
NUM = r"\d+(?:-?[A-Za-z]{1,2}(?![A-Za-z]))?"
# separators between numbers: ", " / ", and " / " and " / " or " / " to " / " & "   (Oxford comma included)
SEPR = r"(?:\s*,\s*(?:and|or)\s+|\s*,\s*|\s+(?:and|or|to)\s+|\s*&\s*)"
SEC = re.compile(
    rf"(?:\bsections?|\bsecs?\.?|\bss?\.|(?<=under)ss?\.)\s*{NUM}(?:\s*\([^)]*\))*"
    rf"(?:{SEPR}(?:sections?\s*|ss?\.\s*)?{NUM}(?:\s*\([^)]*\))*)*", re.I)
ART = re.compile(
    rf"\barticles?\s+{NUM}(?:\s*\([^)]*\))*(?:{SEPR}{NUM}(?:\s*\([^)]*\))*)*", re.I)

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def _nums(span):
    span = re.sub(r"\([^)]*\)", " ", span)
    out = []
    for n in re.findall(NUM, span):
        n = n.replace("-", "").upper()
        if re.fullmatch(r"\d{4}", n) and 1800 <= int(n) <= 2100:
            continue                                  # a year, not a section
        out.append(n)
    return out

def _slug(words, year):
    toks = [w for w in re.findall(r"[A-Za-z]+", words) if w.lower() not in STOP]
    return "_".join(t.upper() for t in toks[-2:] + ["ACT", year])

def extract_anchors(text):
    acts, secs, pending = set(), set(), []
    for sent in SENT.split(text or ""):
        found = []
        for canon, rx in ALIAS_RX:
            found += [(m.start(), m.end(), canon) for m in rx.finditer(sent)]
        found += [(m.start(), m.end(), _slug(m.group(1), m.group(2))) for m in GENERIC.finditer(sent)]
        for m in ART.finditer(sent):
            acts.add("ACT:CONST")
            secs.update(f"SEC:CONST:{n}" for n in _nums(m.group(0)))
        for m in SEC.finditer(sent):
            nums = _nums(m.group(0))
            if not nums:
                continue
            best, bd = None, 10**9
            for a, b, c in found:
                d = a - m.end() if a >= m.end() else (m.start() - b + 5 if b <= m.start() else 0)
                if d < bd:
                    best, bd = c, d
            if best:
                secs.update(f"SEC:{best}:{n}" for n in nums)
            else:
                pending.append(nums)
        acts.update(f"ACT:{c}" for _, _, c in found)
    if len(acts) == 1 and pending:
        a = next(iter(acts))[4:]
        for nums in pending:
            secs.update(f"SEC:{a}:{n}" for n in nums)
    unattributed = 0 if len(acts) == 1 else len(pending)
    return {"acts": sorted(acts), "secs": sorted(secs),
            "all": sorted(acts | secs), "n_unattributed": unattributed}
