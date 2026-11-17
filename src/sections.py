import re

HEAD = re.compile(r"(?m)^[ \t\*#]*([A-Z][A-Za-z ()/\-]{2,60}?)[ \t\*]*:[ \t]*")

def canon(h):
    h = h.lower().strip()
    if h.startswith("fact"):                         return "facts"
    if h.startswith("issue") or "main issues" in h:  return "issues"
    if h.startswith("statute"):                      return "statute"
    if h.startswith("ratio"):                        return "ratio"
    if h.startswith("ruling") or h.startswith("decision"): return "ruling"
    if h.startswith("conclusion"):                   return "conclusion"
    return None

def split_sections(t):
    ms = [(m.start(), m.end(), canon(m.group(1))) for m in HEAD.finditer(t)]
    ms = [m for m in ms if m[2]]
    out = {}
    for i, (s, e, c) in enumerate(ms):
        end = ms[i + 1][0] if i + 1 < len(ms) else len(t)
        out[c] = (out.get(c, "") + " " + t[e:end].strip()).strip()
    return out

def parse_case(t):
    """query_text = what the system may see (facts+issues+statute).
       ratio / ruling_text are kept separately and NEVER shown for a test case."""
    s = split_sections(t)
    q = " ".join(s.get(k, "") for k in ("facts", "issues", "statute")).strip()
    r = " ".join(s.get(k, "") for k in ("ruling", "conclusion")).strip()
    ok = all(k in s for k in ("facts", "issues", "statute")) and bool(r)
    return {"query_text": q, "ratio": s.get("ratio", ""), "ruling_text": r, "ok": ok}
