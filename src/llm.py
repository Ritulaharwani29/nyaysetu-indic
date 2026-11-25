import os, json, re, time, math
import numpy as np

P = "data/processed"
OUT = "results/stage11_verdicts.jsonl"
EXPL = "results/stage11_explanations.jsonl"
MODEL = "Qwen/Qwen2.5-7B-Instruct"       # open weights; NOT verified for Marathi/Gujarati quality
LANG_NAME = {"en": "English", "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati"}
SCRIPT = {"en": (0x41, 0x7A), "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "gu": (0x0A80, 0x0AFF)}
K = 3                  # precedents in the prompt
PREC_WORDS = 150       # each precedent is cut to this many words (its Facts/Issues/Statute text)
MAX_Q_TOKENS = 1200    # the query is cut to this many tokens of the LLM tokenizer
CONDS = ["none", "cos", "hyb"]
WORD = {1: "ALLOWED", 0: "DISMISSED"}


def trunc_words(t, n):
    w = (t or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def build_user_prompt(query, precs, lang):
    """precs = [(text, label_int), ...]. Instructions stay in English; the query is in its own language."""
    s = ("Task: predict the outcome of an Indian Supreme Court appeal/petition from the case description below.\n"
         "Reply with exactly one word: ALLOWED or DISMISSED.\n")
    if precs:
        s += "\nSimilar past cases (English) with their known outcomes:\n"
        for i, (t, y) in enumerate(precs, 1):
            s += f"\n[{i}] {trunc_words(t, PREC_WORDS)}\nOutcome of [{i}]: {WORD[y]}\n"
    s += f"\nCurrent case ({LANG_NAME[lang]}):\n{query}\n\nOutcome of the current case (ALLOWED or DISMISSED):"
    return s


def build_explain_prompt(query, precs, lang, verdict):
    s = build_user_prompt(query, precs, lang).rsplit("\n\nOutcome of the current case", 1)[0]
    return (s + f"\n\nThe predicted outcome is {WORD[verdict]}. In {LANG_NAME[lang]} only, explain this prediction in 3 to 4 sentences. "
            + ("Refer to the similar cases by number where they matter. " if precs else "") + f"Write the whole answer in {LANG_NAME[lang]}.")


def script_share(text, lang):
    lo, hi = SCRIPT[lang]; letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi or (lang == "en" and 0x61 <= ord(c) <= 0x7A) for c in letters) / len(letters)


def _jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_context(enc, top_path="results/stage10_top.json"):
    """Everything the LLM step needs, from files only: queries, gold labels, split, retrieved ids, corpus text and labels."""
    T = json.load(open(top_path)); best = T["alpha"][enc]
    q = {l: {str(r["document_id"]): r for r in _jl(f"{P}/queries_{l}.jsonl")} for l in ["en", "hi", "mr", "gu"]}
    qids = [str(r["document_id"]) for r in _jl(f"{P}/queries_en.jsonl")]
    corpus = {}
    for r in _jl(f"{P}/corpus_en.jsonl"):
        lab = r.get("label"); lab = str(int(float(lab))) if lab not in (None, "None", "") and lab == lab else ""
        corpus[str(r["document_id"])] = (r["query_text"] or "", int(lab) if lab in ("0", "1") else None)
    return dict(enc=enc, alpha=best, top=T["top"][enc], q=q, qids=qids, corpus=corpus,
                gold={i: int(float(q["en"][i]["label"])) for i in qids}, split={i: str(q["en"][i].get("split")) for i in qids})


def precedents(C, cond, lang, qid):
    if cond == "none":
        return []
    ids = C["top"][lang]["cos" if cond == "cos" else "hyb_ext"][qid][:K]
    out = []
    for i in ids:
        t, y = C["corpus"][i]
        assert y in (0, 1), f"retrieved case {i} has no binary label"
        out.append((t, y))
    return out


def run_verdicts(llm, C, conds, langs, splits=("dev", "test"), limit=None, out=OUT):
    """Resumable: finished (cond, lang, qid) rows are skipped. One forward pass per prompt (no generation)."""
    done = set()
    if os.path.exists(out):
        for r in _jl(out):
            done.add((r["cond"], r["lang"], r["qid"]))
    ids = [i for sp in splits for i in C["qids"] if C["split"][i] == sp]
    if limit:                                     # pilot: the first limit/len(splits) queries of each split
        ids = [i for sp in splits for i in [j for j in C["qids"] if C["split"][j] == sp][: max(1, limit // len(splits))]]
    jobs = [(c, l, i) for c in conds for l in langs for i in ids if (c, l, i) not in done]
    print(f"{len(jobs)} prompts to run ({len(done)} already done)")
    t0 = time.time(); n = 0
    with open(out, "a", encoding="utf-8") as f:
        for c, l, i in jobs:
            q = llm.trunc_query(C["q"][l][i]["query_text"], MAX_Q_TOKENS)
            prompt = llm.chat_prompt(build_user_prompt(q, precedents(C, c, l, i), l))
            p, mass, ntok = llm.verdict(prompt)
            f.write(json.dumps({"cond": c, "lang": l, "qid": i, "split": C["split"][i], "gold": C["gold"][i],
                                "p_allowed": round(p, 5), "mass": round(mass, 4), "ntok": ntok}) + "\n"); f.flush()
            n += 1
            if n in (1, 5, 20) or n % 100 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} | {el / n:.1f} s/prompt | ~{(len(jobs) - n) * el / n / 60:.0f} min left | last ntok {ntok}")
    print("done")


def explain_some(llm, C, df, cond, langs, n_per_lang=15, out=EXPL, max_new=220):
    done = {(r["cond"], r["lang"], r["qid"]) for r in _jl(out)} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for l in langs:
            sub = df[(df.cond == cond) & (df.lang == l) & (df.split == "test")].head(n_per_lang)
            for _, r in sub.iterrows():
                if (cond, l, r.qid) in done:
                    continue
                q = llm.trunc_query(C["q"][l][r.qid]["query_text"], MAX_Q_TOKENS)
                v = int(r.p_allowed > 0.5)
                txt = llm.generate(llm.chat_prompt(build_explain_prompt(q, precedents(C, cond, l, r.qid), l, v)), max_new)
                f.write(json.dumps({"cond": cond, "lang": l, "qid": r.qid, "verdict": WORD[v], "gold": WORD[int(r.gold)],
                                    "script_share": round(script_share(txt, l), 3), "text": txt}, ensure_ascii=False) + "\n"); f.flush()


def macro_f1(g, p):
    f = []
    for c in (0, 1):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum()); fn = int(((p != c) & (g == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df):
    """Per (cond, lang) on TEST: accuracy at 0.5 and at a DEV-calibrated threshold (dev median of p_allowed), macro-F1,
    share predicted ALLOWED, bootstrap CI, paired difference to the no-retrieval run of the same language."""
    import pandas as pd
    rows = []
    for (c, l), g in df.groupby(["cond", "lang"]):
        te, dv = g[g.split == "test"], g[g.split == "dev"]
        if len(te) == 0:
            continue
        gold = te.gold.values; thr = float(dv.p_allowed.median()) if len(dv) >= 10 else 0.5
        p05 = (te.p_allowed.values > 0.5).astype(int); pc = (te.p_allowed.values > thr).astype(int)
        ok05 = (p05 == gold).astype(float)
        row = dict(cond=c, lang=l, n=len(te), acc_at_0_5=round(ok05.mean(), 3), ci="%.3f-%.3f" % boot_ci(ok05),
                   share_allowed=round(p05.mean(), 3), thr_dev=round(thr, 3), acc_dev_thr=round(float((pc == gold).mean()), 3),
                   f1_at_0_5=round(macro_f1(gold, p05), 3), mass_min=round(float(te.mass.min()), 3))
        base = df[(df.cond == "none") & (df.lang == l) & (df.split == "test")].set_index("qid")
        if c != "none" and len(base):
            b = (base.loc[te.qid, "p_allowed"].values > 0.5).astype(int) == gold
            d = ok05 - b.astype(float)
            row["delta_vs_none"] = "%+.3f [%.3f, %.3f]" % ((d.mean(),) + boot_ci(d))
        rows.append(row)
    return pd.DataFrame(rows)


class LLM:
    """Thin wrapper around a Hugging Face chat model on the GPU(s). One forward pass per verdict (no sampling)."""

    def __init__(self, name=MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.name = torch, name
        self.tok = AutoTokenizer.from_pretrained(name)
        n = torch.cuda.device_count()
        assert n >= 1, "no GPU"
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto", max_memory={i: "13GiB" for i in range(n)}).eval()
        a = self.tok.encode("ALLOWED", add_special_tokens=False); d = self.tok.encode("DISMISSED", add_special_tokens=False)
        self.ida, self.idd = a[0], d[0]
        assert self.ida != self.idd, "the first tokens of ALLOWED and DISMISSED are the same -> change the answer words"
        print(f"loaded {name} | first tokens: ALLOWED={self.tok.decode([self.ida])!r} DISMISSED={self.tok.decode([self.idd])!r}")

    def chat_prompt(self, user):
        msgs = [{"role": "system", "content": "You are a careful legal analyst of Indian Supreme Court judgments."},
                {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def trunc_query(self, text, n):
        ids = self.tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= n else self.tok.decode(ids[:n])

    def verdict(self, prompt):
        torch = self.torch
        x = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            try:
                lg = self.model(**x, logits_to_keep=1).logits
            except TypeError:
                lg = self.model(**x).logits
        lp = torch.log_softmax(lg[0, -1].float(), dim=-1)
        a, d = lp[self.ida].item(), lp[self.idd].item()
        return 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, d - a)))), math.exp(a) + math.exp(d), int(x["input_ids"].shape[1])

    def generate(self, prompt, max_new=220):
        torch = self.torch
        x = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            o = self.model.generate(**x, max_new_tokens=max_new, do_sample=False)
        return self.tok.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
