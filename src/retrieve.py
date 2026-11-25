import os, json, collections
import numpy as np
import pandas as pd
import scipy.sparse as sp

P = "data/processed"
EMB = "/kaggle/working/emb"
LANGS = ["en", "hi", "mr", "gu"]
TRANS = ["hi", "mr", "gu"]
ENCS = ["bge_m3", "e5_large_instruct", "e5_base", "labse"]
ALPHAS = [1.0, 0.95, 0.9, 0.8, 0.6]      # 1.0 = pure cosine
POOL = 100                               # rerank pool = top-100 by cosine
KS = [3, 5, 9]                           # odd k, so a binary vote never ties
WEAK_W = 0.5                             # weight of a weak (number-only) anchor
STRATA = ["all", "section", "act_only", "none"]


def _lab(v):
    if v is None or v != v:
        return ""
    try:
        s = str(int(float(v)))
    except (TypeError, ValueError):
        s = str(v).strip()
    return s if s in ("0", "1", "2") else ""


def _jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class AnchorIndex:
    """IDF-weighted anchor overlap between a query and corpus documents (invariant anchors)."""

    def __init__(self, doc_anchor_lists):
        n = len(doc_anchor_lists); self.n = n
        sets = [set(l) for l in doc_anchor_lists]
        nums = [set(a.rsplit(":", 1)[1] for a in s if a.startswith("SEC:")) for s in sets]
        self.v, self.idf, self.D = self._build(sets, n)
        self.v2, self.idf2, self.D2 = self._build(nums, n)      # section NUMBERS only (for weak anchors)
        self.unk = float(np.log(n + 1))                          # idf of an anchor no document has

    @staticmethod
    def _build(sets, n):
        df = collections.Counter(a for s in sets for a in s)
        v = {a: i for i, a in enumerate(df)}
        rows, cols = [], []
        for i, s in enumerate(sets):
            for a in s:
                rows.append(i); cols.append(v[a])
        V = max(1, len(v))
        D = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, V))
        idf = np.zeros(V, np.float32)
        for a, i in v.items():
            idf[i] = np.log((n + 1) / (df[a] + 1))
        return v, idf, D

    def overlap(self, ext, weak, rows):
        """sum idf(shared anchors) / sum idf(query anchors), in. No anchors -> 0 for every document."""
        w = np.zeros(self.D.shape[1], np.float32); w2 = np.zeros(self.D2.shape[1], np.float32); den = 0.0
        for a in ext:
            i = self.v.get(a)
            if i is None:
                den += self.unk
            else:
                w[i] = self.idf[i]; den += float(self.idf[i])
        for a in weak:
            i = self.v2.get(a.rsplit(":", 1)[1])
            if i is None:
                den += WEAK_W * self.unk
            else:
                w2[i] = WEAK_W * self.idf2[i]; den += WEAK_W * float(self.idf2[i])
        if den <= 0:
            return np.zeros(len(rows), np.float32)
        num = self.D[rows] @ w + self.D2[rows] @ w2
        return (num / den).astype(np.float32)


def load_data():
    ids = [str(i) for i in json.load(open(f"{EMB}/corpus_ids.json"))]
    lab = {str(r["document_id"]): _lab(r.get("label")) for r in _jl(f"{P}/corpus_en.jsonl")}
    inv = {str(r["document_id"]): r["inv"] for r in _jl(f"{P}/anchors_corpus_en.jsonl")}
    labels = np.array([lab.get(i, "") for i in ids])
    keep = np.where((labels == "0") | (labels == "1"))[0]          # the search index = binary-labelled cases
    ids_l = [ids[i] for i in keep]
    dlab = labels[keep].astype(int)
    doc_inv = [inv.get(i, []) for i in ids_l]
    qen = _jl(f"{P}/queries_en.jsonl")
    qids = [str(r["document_id"]) for r in qen]
    assert not (set(qids) & set(ids)), "held-out ids found inside the corpus"
    glab = [_lab(r["label"]) for r in qen]
    assert all(g in ("0", "1") for g in glab), "a query has no binary label"
    gold = np.array([int(g) for g in glab])
    split = np.array([str(r.get("split")) for r in qen])
    # anchors are recomputed from the query texts with the Stage 8 extractors (no dependence on the anchors_queries_*.jsonl columns)
    from src.anchors import extract_anchors
    from src.anchors_indic import extract_anchors_indic, invariant_set
    qtext = {l: {str(r["document_id"]): r["query_text"] for r in _jl(f"{P}/queries_{l}.jsonl")} for l in LANGS}
    oracle = [set(invariant_set(extract_anchors(qtext["en"][i])["all"])) for i in qids]
    ext, weak = {"en": oracle}, {"en": [set() for _ in qids]}
    for l in TRANS:
        e, w = [], []
        for i in qids:
            r = extract_anchors_indic(qtext[l][i])
            a_ = set(r["all"]); wk = set(r["weak"]) | {x for x in a_ if ":?:" in x}
            e.append({x for x in a_ if ":?:" not in x}); w.append(wk)
        ext[l], weak[l] = e, w
    osec = [{a for a in o if a.startswith("SEC:")} for o in oracle]
    stratum = np.array(["section" if s else ("act_only" if o else "none") for o, s in zip(oracle, osec)])
    return dict(ids_l=ids_l, dlab=dlab, keep=keep, index=AnchorIndex(doc_inv), doc_sets=[set(x) for x in doc_inv],
                qids=qids, gold=gold, split=split, ext=ext, weak=weak, oracle=oracle, osec=osec, stratum=stratum)


def load_q(enc, lang, qids):
    ids = [str(i) for i in json.load(open(f"{EMB}/query_ids_{lang}.json"))]
    arr = np.load(f"{EMB}/{enc}/q_{lang}.npy").astype(np.float32)
    pos = {d: i for i, d in enumerate(ids)}
    return arr[[pos[i] for i in qids]]


def pools(Cl, Qv):
    S = Qv @ Cl.T
    idx = np.argpartition(-S, POOL, axis=1)[:, :POOL]
    sc = np.take_along_axis(S, idx, 1)
    o = np.argsort(-sc, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(sc, o, 1)


def overlaps(index, pool_idx, ext_list, weak_list):
    return np.stack([index.overlap(e, w, pool_idx[i]) for i, (e, w) in enumerate(zip(ext_list, weak_list))])


def rerank(pool_idx, pool_cos, ov, alpha, k=max(KS)):
    score = alpha * pool_cos + (1 - alpha) * ov
    o = np.argsort(-score, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(pool_idx, o, 1)


def build_all(D, encs=None, seed=0):
    """TOP[(encoder, lang, strategy, alpha)] -> (n_queries, 9) indices into the labelled corpus."""
    TOP = {}
    n, nq = len(D["ids_l"]), len(D["qids"])
    rng = np.random.default_rng(seed)
    TOP[("-", "-", "random", 0.0)] = np.stack([rng.choice(n, max(KS), replace=False) for _ in range(nq)])
    for enc in (encs or ENCS):
        fn = f"{EMB}/{enc}/corpus.npy"
        if not os.path.exists(fn):
            print(enc, "-> embeddings missing, skipped"); continue
        Cl = np.load(fn)[D["keep"]].astype(np.float32)
        for lang in LANGS:
            pi, pc = pools(Cl, load_q(enc, lang, D["qids"]))
            ov_ext = overlaps(D["index"], pi, D["ext"][lang], D["weak"][lang])
            ov_ora = overlaps(D["index"], pi, D["oracle"], [set()] * nq) if lang != "en" else None
            TOP[(enc, lang, "cos", 1.0)] = rerank(pi, pc, ov_ext, 1.0)
            for a in ALPHAS[1:]:
                TOP[(enc, lang, "hyb_ext", a)] = rerank(pi, pc, ov_ext, a)
                if ov_ora is not None:
                    TOP[(enc, lang, "hyb_oracle", a)] = rerank(pi, pc, ov_ora, a)
        del Cl
        print(enc, "done")
    return TOP


def macro_f1(g, p):
    f = []
    for c in (0, 1):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum()); fn = int(((p != c) & (g == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def knn_pred(top, D, k):
    return (D["dlab"][top[:, :k]].sum(1) * 2 > k).astype(int)


def sec_hit(top, D, k=5):
    """1 if any of the top-k documents shares a SECTION anchor with the English (oracle) query; nan if the query has none."""
    out = np.full(len(top), np.nan)
    for i, (row, qsec) in enumerate(zip(top[:, :k], D["osec"])):
        if qsec:
            out[i] = float(any(bool(qsec & D["doc_sets"][j]) for j in row))
    return out


def selftest():
    idx = AnchorIndex([["ACT:IPC", "SEC:IPC:302"], ["ACT:CRPC"]])
    ov = idx.overlap({"SEC:IPC:302"}, {"SEC:?:302"}, np.array([0, 1]))
    assert ov.shape == (2,) and ov[0] > ov[1] == 0.0
    print("READY")
    return True


def table(TOP, D):
    rows = []
    for (enc, lang, strat, alpha), top in TOP.items():
        sh = sec_hit(top, D, k=5)
        for sp_name in ["dev", "test"]:
            m_sp = (D["split"] == sp_name)
            for st in STRATA:
                m = m_sp if st == "all" else (m_sp & (D["stratum"] == st))
                if not m.any():
                    continue
                g = D["gold"][m]
                row = dict(
                    encoder=enc, lang=lang, strategy=strat, alpha=alpha,
                    split=sp_name, stratum=st, n=int(m.sum()),
                    sec_hit5=float(np.nanmean(sh[m])) if np.isfinite(sh[m]).any() else np.nan
                )
                for k in KS:
                    p = knn_pred(top[m], D, k)
                    row[f"acc{k}"] = float((p == g).mean())
                    row[f"f1_{k}"] = macro_f1(g, p)
                rows.append(row)
    return pd.DataFrame(rows)


def choose_alpha(df):
    sub = df[(df["split"] == "dev") & (df["stratum"] == "all") & (df["strategy"] == "hyb_ext")]
    best = {}
    for enc, grp in sub.groupby("encoder"):
        means = grp.groupby("alpha")["acc5"].mean()
        best[enc] = float(means.idxmax())
    return best


def _boot(arr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def headline(TOP, D, best):
    m = (D["split"] == "test")
    g = D["gold"][m]
    rows = []
    for enc in ENCS:
        if enc not in best:
            continue
        a = best[enc]
        for lang in LANGS:
            if (enc, lang, "cos", 1.0) not in TOP:
                continue
            c_cos = (knn_pred(TOP[(enc, lang, "cos", 1.0)][m], D, 5) == g).astype(float)
            c_hyb = (knn_pred(TOP[(enc, lang, "hyb_ext", a)][m], D, 5) == g).astype(float)
            lo_c, hi_c = _boot(c_cos)
            lo_h, hi_h = _boot(c_hyb)
            lo_d, hi_d = _boot(c_hyb - c_cos)
            rows.append(dict(
                encoder=enc, lang=lang, alpha=a,
                cos_acc5=f"{c_cos.mean():.3f} [{lo_c:.3f}, {hi_c:.3f}]",
                hyb_acc5=f"{c_hyb.mean():.3f} [{lo_h:.3f}, {hi_h:.3f}]",
                diff=f"{(c_hyb - c_cos).mean():+.3f} [{lo_d:+.3f}, {hi_d:+.3f}]",
                sec_hit5_cos=round(float(np.nanmean(sec_hit(TOP[(enc, lang, "cos", 1.0)][m], D, 5))), 3),
                sec_hit5_hyb=round(float(np.nanmean(sec_hit(TOP[(enc, lang, "hyb_ext", a)][m], D, 5))), 3),
            ))
    return pd.DataFrame(rows)


def dump_top(TOP, D, best, path):
    out = {"alpha": {k: float(v) for k, v in best.items()}, "top": {}}
    for enc in ENCS:
        if enc not in best:
            continue
        a = best[enc]
        out["top"][enc] = {}
        for lang in LANGS:
            out["top"][enc][lang] = {}
            for strat, al in [("cos", 1.0), ("hyb_ext", a), ("hyb_oracle", a)]:
                if (enc, lang, strat, al) not in TOP:
                    continue
                top = TOP[(enc, lang, strat, al)]
                out["top"][enc][lang][strat] = {
                    qid: [D["ids_l"][j] for j in row]
                    for qid, row in zip(D["qids"], top)
                }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
