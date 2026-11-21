"""Recreate data/processed/corpus_en.jsonl from raw NyayaRAG + the saved split ids.

    python scripts/rebuild_corpus.py

Deterministic: with the saved ids it must reproduce the Stage 3 corpus (52,947 rows)."""
import json, os, sys, zipfile, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.sections import parse_case

RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
HF_REPO = "L-NLProc/NyayaRAG"
NEED = {                                    # zip-name keyword -> the one file we need from that zip
    "Base": "updated_SCI_56k_single.json",                       # labels
    "Previous_Similar": "SCI_judgements_56k_summarised.json",    # 54,782 summaries (retrieval corpus)
}
EXPECTED_ROWS = 52947                       # Stage 3 result, valid when no extra held-out ids exist


def fetch(keyword, member):
    out = RAW / "extracted" / member
    if out.exists():
        print("already extracted:", member)
        return out
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    files = HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=tok)
    name = next(f for f in files if keyword in f and f.endswith(".zip"))
    print("downloading", name)
    z = hf_hub_download(HF_REPO, name, repo_type="dataset", token=tok, local_dir=RAW / "_zips")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        m = next(n for n in zf.namelist() if n.endswith("/" + member) or n == member)
        with zf.open(m) as src, open(out, "wb") as dst:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                dst.write(chunk)
    return out


def load_records(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else list(d.values())


def build_corpus(sim_json, labels, held):
    df = pd.DataFrame(load_records(sim_json))
    df["document_id"] = df.document_id.astype(str)
    p = pd.DataFrame(list(df.summarized_text.astype(str).apply(parse_case)), index=df.index)
    df = pd.concat([df[["document_id"]], p], axis=1)
    n_all = len(df)
    df = df[df.ok & ~df.document_id.isin(held)].copy()
    df["label"] = df.document_id.map(labels)           # NaN where the base file has no label
    return df[["document_id", "label", "query_text", "ruling_text"]], n_all


def main():
    split = json.load(open(PROC / "split_ids.json"))
    held = set(map(str, split["test"])) | set(map(str, split["dev"]))
    extra = PROC / "split_ids_extra.json"            # optional: more held-out ids added later
    has_extra = extra.exists()
    if has_extra:
        held |= set(map(str, json.load(open(extra))["ids"]))
    print("held-out ids:", len(held), "(with extra)" if has_extra else "")

    base = fetch("Base", NEED["Base"])
    sim = fetch("Previous_Similar", NEED["Previous_Similar"])

    rec = load_records(base)
    labels = {str(r["document_id"]): str(r["label"]) for r in rec}
    del rec
    print("labels loaded:", len(labels))

    corpus, n_all = build_corpus(sim, labels, held)
    PROC.mkdir(parents=True, exist_ok=True)
    corpus.to_json(PROC / "corpus_en.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"\nsummaries in file: {n_all} | kept (parsed ok, not held-out): {len(corpus)}")
    print("with a label:", corpus.label.notna().sum(),
          "| binary (0/1):", corpus.label.isin(["0", "1"]).sum())
    print("held-out ids inside corpus (must be 0):", len(held & set(corpus.document_id)))
    if not has_extra:
        print("matches Stage 3 corpus size:", "YES" if len(corpus) == EXPECTED_ROWS
              else f"NO (expected {EXPECTED_ROWS})")


if __name__ == "__main__":
    main()
