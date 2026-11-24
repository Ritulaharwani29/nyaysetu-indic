import os, json, gc, sys, time
import numpy as np

P = "data/processed"
EMB = "/kaggle/working/emb"
CHROMA = "/kaggle/working/chroma"
LANGS = ["en", "hi", "mr", "gu"]
MAXLEN = 512
CHUNK = 2048

# e5-large-instruct: instruction on the QUERY side only, nothing on passages.
# e5-base: "query: " / "passage: ".  bge-m3 and LaBSE: no prefixes.
ENCODERS = {
    "bge_m3": dict(name="BAAI/bge-m3", q_prefix="", p_prefix="", bs=16),
    "e5_large_instruct": dict(
        name="intfloat/multilingual-e5-large-instruct",
        q_prefix="Instruct: Given a legal case description, retrieve similar Indian Supreme Court case summaries\nQuery: ",
        p_prefix="", bs=16),
    "e5_base": dict(name="intfloat/multilingual-e5-base", q_prefix="query: ", p_prefix="passage: ", bs=64),
    "labse": dict(name="sentence-transformers/LaBSE", q_prefix="", p_prefix="", bs=64),
}


def _lab(v):
    """1, 1.0, "1", None, NaN, "None" -> "0"/"1"/"2" or ""."""
    if v is None or v != v:
        return ""
    try:
        s = str(int(float(v)))
    except (TypeError, ValueError):
        s = str(v).strip()
    return s if s in ("0", "1", "2") else ""


def load_corpus():
    ids, texts, labels = [], [], []
    with open(f"{P}/corpus_en.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                ids.append(str(r["document_id"])); texts.append(r["query_text"] or ""); labels.append(_lab(r.get("label")))
    return ids, texts, labels


def load_queries(lang):
    ids, texts = [], []
    with open(f"{P}/queries_{lang}.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                ids.append(str(r["document_id"])); texts.append(r["query_text"] or "")
    return ids, texts


def load_model(key):
    import torch
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -> check Accelerator = GPU")
    m = SentenceTransformer(ENCODERS[key]["name"], device="cuda")
    m.max_seq_length = MAXLEN
    m.half()
    return m


def encode_texts(model, texts, prefix, bs):
    """Normalised float16 embeddings. On out-of-memory the batch size is halved and the call retried."""
    import torch
    inp = [prefix + t for t in texts]
    while True:
        try:
            v = model.encode(inp, batch_size=bs, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            return v.astype(np.float16)
        except torch.cuda.OutOfMemoryError:
            pass
        gc.collect(); torch.cuda.empty_cache()
        if bs == 1:
            raise RuntimeError("out of memory even with batch size 1")
        bs = max(1, bs // 2)
        print(f"  out of memory -> batch size {bs}")


def _save(path, arr):
    tmp = os.path.join(os.path.dirname(path), ".tmp.npy")
    np.save(tmp, arr); os.replace(tmp, path)             # a half-written file never has the final name


def run_encoder(key):
    """Queries (4 languages) first, then the corpus in chunks. Safe to re-run: finished files are skipped."""
    import torch
    cfg = ENCODERS[key]; out = f"{EMB}/{key}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/corpus.npy") and all(os.path.exists(f"{out}/q_{l}.npy") for l in LANGS):
        print(key, "already done"); return
    ids, texts, _ = load_corpus()
    json.dump(ids, open(f"{EMB}/corpus_ids.json", "w"))
    model = load_model(key)
    print(f"{key}: {cfg['name']} | max_seq_length {model.max_seq_length} | batch {cfg['bs']} | device {model.device}")
    for lang in LANGS:
        fn = f"{out}/q_{lang}.npy"
        if os.path.exists(fn):
            continue
        qids, qt = load_queries(lang)
        _save(fn, encode_texts(model, qt, cfg["q_prefix"], cfg["bs"]))
        json.dump(qids, open(f"{EMB}/query_ids_{lang}.json", "w"))
        print("  queries", lang, "done")
    n = len(texts); nch = (n + CHUNK - 1) // CHUNK; t0 = time.time(); first = True
    for i in range(nch):
        fn = f"{out}/corpus_{i:03d}.npy"
        if os.path.exists(fn):
            continue
        s, e = i * CHUNK, min(n, (i + 1) * CHUNK); t1 = time.time()
        _save(fn, encode_texts(model, texts[s:e], cfg["p_prefix"], cfg["bs"]))
        rate = (e - s) / (time.time() - t1)
        print(f"  chunk {i + 1}/{nch} | {rate:.1f} texts/s | elapsed {(time.time() - t0) / 60:.1f} min"
              + (f" | rough total {n / rate / 60:.0f} min" if first else ""))
        first = False
    arr = np.concatenate([np.load(f"{out}/corpus_{i:03d}.npy") for i in range(nch)])
    assert arr.shape[0] == n and not np.isnan(arr.astype(np.float32)).any()
    _save(f"{out}/corpus.npy", arr)
    for i in range(nch):
        os.remove(f"{out}/corpus_{i:03d}.npy")
    print(f"{key}: saved corpus {arr.shape} {arr.dtype}")
    del model; gc.collect(); torch.cuda.empty_cache()


def alignment(key, k=5):
    """Cross-lingual check on the 360 queries: does the hi/mr/gu query land next to its English twin?"""
    C = np.load(f"{EMB}/{key}/corpus.npy").astype(np.float32)
    Q = {l: np.load(f"{EMB}/{key}/q_{l}.npy").astype(np.float32) for l in LANGS}

    def topk(M):
        S = M @ C.T
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        o = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        return np.take_along_axis(idx, o, 1)

    T_en = topk(Q["en"]); rows = []
    for l in LANGS[1:]:
        sim = Q[l] @ Q["en"].T; n = len(sim); diag = np.diag(sim)
        T = topk(Q[l])
        rows.append({"encoder": key, "lang": l,
                     "twin_top1": round(float((sim.argmax(1) == np.arange(n)).mean()), 3),
                     "twin_cos": round(float(diag.mean()), 3),
                     "other_cos": round(float((sim.sum() - diag.sum()) / (n * n - n)), 3),
                     f"top{k}_overlap_with_en": round(float(np.mean([len(set(a) & set(b)) / k for a, b in zip(T_en, T)])), 3),
                     "top1_same_as_en": round(float((T[:, 0] == T_en[:, 0]).mean()), 3)})
    return rows


def corpus_meta(ids, labels):
    inv = {}
    with open(f"{P}/anchors_corpus_en.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); inv[str(r["document_id"])] = r["inv"]
    metas = []
    for i, lab in zip(ids, labels):
        a = inv.get(i, [])
        metas.append({"label": lab, "n_anchors": len(a), "inv": "|".join(a) if a else "-"})
    return metas


def build_chroma(key):
    import chromadb
    ids = json.load(open(f"{EMB}/corpus_ids.json"))
    _, _, labels = load_corpus()
    assert len(set(ids)) == len(ids), "duplicate document_id in the corpus"
    E = np.load(f"{EMB}/{key}/corpus.npy")
    assert E.shape[0] == len(ids)
    metas = corpus_meta(ids, labels)
    held = set(load_queries("en")[0])
    assert not (held & set(ids)), "held-out ids found inside the corpus"
    client = chromadb.PersistentClient(path=f"{CHROMA}/{key}")
    try:
        client.delete_collection("corpus")
    except Exception:
        pass
    col = client.create_collection("corpus", metadata={"hnsw:space": "cosine"})
    B = 4000
    for s in range(0, len(ids), B):
        e = min(len(ids), s + B)
        col.add(ids=ids[s:e], embeddings=E[s:e].astype(np.float32).tolist(), metadatas=metas[s:e])
    assert col.count() == len(ids)
    print(f"{key}: Chroma collection built, {col.count():,} items, dim {E.shape[1]}")
    return col


def chroma_selftest(key, n=5):
    """Querying with a stored vector must return itself at distance ~0, and the 2nd distance must equal 1 - dot (= cosine)."""
    import chromadb
    col = chromadb.PersistentClient(path=f"{CHROMA}/{key}").get_collection("corpus")
    ids = json.load(open(f"{EMB}/corpus_ids.json"))
    pos = {d: i for i, d in enumerate(ids)}
    E = np.load(f"{EMB}/{key}/corpus.npy").astype(np.float32)
    rng = np.random.default_rng(1); ok = True
    for i in rng.choice(len(ids), n, replace=False):
        r = col.query(query_embeddings=[E[i].tolist()], n_results=2, include=["distances"])
        rid, d = r["ids"][0], r["distances"][0]
        j = pos[rid[1]]
        exp = 1 - float(E[i] @ E[j])
        good = rid[0] == ids[i] and d[0] < 0.01 and abs(d[1] - exp) < 0.01
        ok &= good
        print(f"  {ids[i]:>10s} -> self {rid[0] == ids[i]} d0={d[0]:.4f} | d1={d[1]:.4f} vs 1-dot={exp:.4f} {'OK' if good else 'CHECK'}")
    return ok
