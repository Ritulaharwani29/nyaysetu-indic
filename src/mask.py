import re

# Sentences that describe a court deciding an appeal/petition.
VERDICT = re.compile(
    r"\b(dismiss\w*|allow(?:ed|ing)?\s+the\s+(?:appeal|petition|writ)|set\s+aside|"
    r"quash\w*|upheld|uphold\w*|affirm\w*)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")

def mask_outcomes(text, token="[OUTCOME]"):
    """Replace any sentence containing decision wording with one token. Returns (text, n_masked)."""
    out, n = [], 0
    for s in SENT.split(text):
        if VERDICT.search(s):
            n += 1
            if not out or out[-1] != token:
                out.append(token)
        else:
            out.append(s)
    return " ".join(out), n
