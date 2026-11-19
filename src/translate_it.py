import re, gc, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

TOK_ID   = "ai4bharat/indictrans2-en-indic-1B"        # tokenizer (custom code, gated)
MODEL_ID = "law-ai/InLegalTrans-En2Indic-1B"          # legal fine-tune of IndicTrans2
LANG = {"hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_USE_CACHE = True                                      # switches to False automatically if needed
_DIG = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")


def nums_of(s):
    """Sorted list of digit strings in a sentence (native digits converted to ASCII)."""
    return sorted(re.findall(r"\d+", s.translate(_DIG)))


def load(model_id=MODEL_ID, half=True):
    tok = AutoTokenizer.from_pretrained(TOK_ID, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="eager",
        low_cpu_mem_usage=True)
    # This model's custom config has no `vocab_size`, which newer transformers requires in beam search.
    if not hasattr(model.config, "vocab_size"):
        model.config.vocab_size = (getattr(model.config, "decoder_vocab_size", None)
                                   or model.get_output_embeddings().out_features)
    model = model.to(DEVICE)
    if DEVICE == "cuda" and half:
        model = model.half()
    print("model loaded from:", model.name_or_path, "| vocab_size:", model.config.vocab_size)
    return tok, model.eval(), IndicProcessor(inference=True)


def chunks(text, max_chars=450):
    """Sentence-level units (the model is trained on sentences, not paragraphs)."""
    out = []
    for s in SENT.split(text.strip()):
        s = s.strip()
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
            cut = cut + 1 if cut > 100 else max_chars
            out.append(s[:cut].strip()); s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def _generate(model, enc, beams, n_ret):
    """Fast path (KV cache); if the model code is incompatible, fall back once to no cache."""
    global _USE_CACHE
    kw = dict(min_length=0, max_length=256, num_beams=beams, num_return_sequences=n_ret)
    try:
        return model.generate(**enc, use_cache=_USE_CACHE, **kw)
    except Exception as e:
        if _USE_CACHE and "out of memory" not in str(e).lower():
            print("generate failed with use_cache=True:", type(e).__name__, str(e)[:160],
                  "\n-> retrying with use_cache=False (slower)")
            _USE_CACHE = False
            return model.generate(**enc, use_cache=False, **kw)
        raise


def _raw(strs, tok, model, beams, nbest):
    """Tokenise + generate for already pre-processed strings. On GPU out-of-memory the batch is
    split in half and retried. Returns len(strs)*nbest decoded strings, sentence-major order."""
    try:
        enc = tok(strs, truncation=True, padding="longest", return_tensors="pt",
                  return_attention_mask=True).to(DEVICE)
        gen = _generate(model, enc, beams, nbest)
        return tok.batch_decode(gen.detach().cpu().tolist(), skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    except torch.cuda.OutOfMemoryError:
        pass                                  # leave the except block first, so the crashed frames can be freed
    gc.collect(); torch.cuda.empty_cache()
    if len(strs) == 1:
        raise RuntimeError("GPU out of memory even for a single sentence")
    h = len(strs) // 2
    return _raw(strs[:h], tok, model, beams, nbest) + _raw(strs[h:], tok, model, beams, nbest)


@torch.no_grad()
def translate_list(sents, tgt, tok, model, ip, bs=8, beams=5, nbest=4, return_flags=False):
    """Translate sentences. Among the n-best candidates, prefer the first one whose digit strings
    match the English sentence (section numbers, years). Flags: top = best candidate was fine,
    nbest = a lower-ranked candidate fixed the numbers, mismatch = none matched (top-1 kept)."""
    code = LANG[tgt]
    beams = max(beams, nbest)
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))   # similar lengths batch together
    res, flags = [None] * len(sents), [None] * len(sents)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        texts = [sents[j] for j in idx]
        pre = ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=code)
        raw = _raw(pre, tok, model, beams, nbest)
        ranks = []                                   # ranks[r] = r-th best candidate for every sentence
        for r in range(nbest):
            ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=code)   # postprocess consumes the stored placeholder maps
            ranks.append(ip.postprocess_batch(raw[r::nbest], lang=code))
        for k, j in enumerate(idx):
            want = nums_of(sents[j])
            cands = [ranks[r][k] for r in range(nbest)]
            pick, flag = cands[0], ("top" if nums_of(cands[0]) == want else "mismatch")
            if flag == "mismatch":
                for c in cands[1:]:
                    if nums_of(c) == want:
                        pick, flag = c, "nbest"
                        break
            res[j], flags[j] = pick, flag
    return (res, flags) if return_flags else res


def translate_queries(texts, tgt, tok, model, ip, **kw):
    per = [chunks(t) for t in texts]
    out, fl = translate_list([c for p in per for c in p], tgt, tok, model, ip, return_flags=True, **kw)
    res, k = [], 0
    for p in per:
        res.append({"en": p, "tr": out[k:k + len(p)], "flags": fl[k:k + len(p)]}); k += len(p)
    return res
