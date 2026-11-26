# NyaySetu-Indic

**Cross-lingual retrieval-augmented judgment prediction for Indian Supreme Court cases, with queries in Hindi, Marathi and Gujarati over an English case corpus.**

NyaySetu-Indic ("Nyay" = justice, "Setu" = bridge) extends the [NyayaRAG](https://arxiv.org/abs/2508.00709) setting (retrieval-augmented legal judgment prediction) to Indian languages. The cases and the retrieval corpus stay in English; only the **queries** are translated. The project asks whether a Hindi, Marathi or Gujarati case description can retrieve the right English precedents, and whether those precedents help a language model predict the outcome and explain it in the query's language.

> **Status:** completed student research project. The pipeline, benchmark and evaluation are finished. Headline results: **cross-lingual retrieval of statutorily relevant precedents clearly works**; **verdict accuracy at the 0.5 threshold is not significantly better than the majority-class baseline for any system**; threshold-free metrics show a **weak signal** when precedents are retrieved (AUC 0.57-0.63 vs about 0.48 without retrieval). Details, caveats and limitations are below.

---

## 1. Contributions

1. **A multilingual benchmark.** 360 held-out cases (300 test + 60 dev) with queries translated into Hindi, Marathi and Gujarati using a legal-domain translation model, plus sentence-level review sheets for native-speaker checking.
2. **Language-invariant legal anchors.** An extractor that turns section numbers, act years and the big codes (IPC, CrPC, CPC, Constitution) in English, Hindi, Marathi or Gujarati text into canonical IDs, so a query in any language can be matched to English precedents by what they cite.
3. **A cross-lingual retriever.** Multilingual embeddings (four encoders compared) with an optional hybrid rerank: `alpha * cosine + (1 - alpha) * IDF-weighted anchor overlap`, with alpha chosen per encoder on the dev set.
4. **Verdict and reasoning in the query language.** An LLM is given the case plus the top retrieved precedents *with their outcomes* and returns a verdict (from next-token probabilities) and an explanation in the query's language.
5. **Evaluation.** Accuracy, macro-F1, AUC, bootstrap confidence intervals, paired tests, kNN-vote and statutory-relevance retrieval metrics, stratification by anchor availability, and cross-lingual consistency.

## 2. Pipeline

```
case (hi / mr / gu)
  -> extract anchors (kept aside)
  -> multilingual encoder (BGE-M3 / mE5-large-instruct / mE5-base / LaBSE, 512 tokens)
  -> cosine search over the ENGLISH train corpus (held-out cases excluded)
  -> hybrid rerank: alpha * cosine + (1 - alpha) * IDF-weighted anchor overlap   (alpha chosen per encoder on dev from {1.0, 0.95, 0.9, 0.8, 0.6}; bge_m3: 0.6)
  -> LLM prompt: case + top precedents WITH their outcomes
  -> verdict (ALLOWED vs DISMISSED, from next-token probabilities) + explanation in the query language
```

Systems compared in the end-to-end evaluation:

| Code | System | Query |
|---|---|---|
| `none` | LLM with no retrieved precedents | en / hi / mr / gu |
| `cos` | LLM + top precedents by cosine similarity | en / hi / mr / gu |
| `hyb` | LLM + top precedents after the anchor-hybrid rerank | en / hi / mr / gu |

LLM: Qwen2.5-7B-Instruct, K = 3 precedents of 150 words each; verdict from next-token probabilities, explanations by greedy decoding. Precedents come from bge_m3 retrieval (results/stage10_top.json).

## 3. Data

| Item | Value |
|---|---|
| Source | Hugging Face `L-NLProc/NyayaRAG` (Supreme Court summaries) |
| Input cases / retrieval summaries | 4,962 / 54,782 |
| Labels | `0` rejected/dismissed, `1` allowed, `2` partial. Meanings were **inferred from ruling text, not documented**. Only binary 0/1 is used. |
| Query text | **Facts + Issues + Statute only.** Ratio, Ruling and Conclusion are never shown for a test case. |
| Evaluation pool | 3,952 binary cases with at least 80 words; held-out **300 test + 60 dev**, stratified, seed 42 |
| Retrieval corpus | 52,947 English summaries (held-out ids removed), of which **23,883 have a binary label** |
| Test balance | 49.7% allowed; **majority-class baseline = 0.503** |
| Anchor strata of the 360 queries | 166 with a section anchor, 150 act-only, 44 with no anchor |

**Leakage check** (TF-IDF + logistic regression, 5-fold, binary). This is why the query excludes the ruling:

| Input | Accuracy |
|---|---|
| Majority class | 0.508 |
| Facts + issues + statute (our query) | 0.568 |
| + ratio | 0.584 |
| + ruling + conclusion | 0.705 |
| ruling + conclusion only | 0.755 |

Query-only TF-IDF reaches about 0.59 (vs 0.505 baseline), which is the realistic ceiling for outcome prediction from this input. 11.8% of queries still contain verdict-like sentences (mostly lower-court history); masking them changed TF-IDF accuracy only from 0.592 to 0.586, so the original query is the main setting.

## 4. Translation

Queries were translated with `law-ai/InLegalTrans-En2Indic-1B` (a legal fine-tune of IndicTrans2), using beam 5 with a 4-best selection that prefers a candidate whose digits match the English. On a 119-sentence sample, digit-match rates were 0.92 / 0.94 / 0.97 (hi / mr / gu), about ±5 points. Observed hazards: the translator sometimes **changes section numbers** (302 -> 303 in Marathi and Gujarati) and **drops year ranges** (1947-48). These errors flow straight into anchor extraction and are reported as findings, not hidden.

## 5. Legal anchors

Act names do not survive translation (Hindi translates "अधिनियम", Marathi and Gujarati often transliterate "एक्ट"). What survives is section numbers, act years and the big codes. Both sides are therefore reduced to an invariant vocabulary:

`ACT:IPC / CRPC / CPC / CONST`, `ACT:Y1922` (any other named act, keyed by year), `SEC:IPC:302`, `SEC:CONST:21`, `SEC:Y1922:16`, plus weak anchors `SEC:?:342` when the number is known but the act is not (Marathi and Gujarati use one word for Section and Article).

Agreement with the anchors of the English original (360 queries), final extractor v4 (`results/stage8_anchor_eval.csv`).

| | hi | mr | gu |
|---|---|---|---|
| Act recall | 0.984 | 0.991 | 0.966 |
| Act precision | 0.977 | 0.980 | 0.962 |
| Section recall, strict | 0.944 | 0.879 | 0.859 |
| Section precision, strict | 0.889 | 0.873 | 0.809 |
| Section recall, number only (weak counted) | 0.970 | 0.920 | 0.910 |
| Exact anchor-set match (316 queries with an anchor) | 0.892 | 0.854 | 0.791 |

Most remaining misses come from the translator, not the extractor: the section number is absent from the translation for 8 / 19 / 18 anchors and changed to another number for 11 / 19 / 23 extra anchors, while genuine extractor gaps are 1 / 5 / 9 (`results/stage8_anchor_why.csv`).

![Fig 2](results/figures/fig2_anchor_quality.png)

Corpus check: collapsing named acts to their year reduced 12,920 distinct anchors to 6,296 while keeping the outcome signal of the section-anchor vote (0.570 full vs 0.571 invariant, against a baseline of about 0.53), and 163 of 191 act-years are shared by more than one act. The invariant form is therefore coarser but not harmful. 86.2% of binary-labelled corpus cases carry at least one anchor.

The extractor has a 24-case self-test (`24 of 24 passed | READY`) that must pass before any evaluation. After v4 the tuning was frozen: remaining differences are reported, not chased.

## 6. Results

![Fig 1](results/figures/fig1_retrieval_statutory_relevance.png)

### 6.1 Retrieval: does a Hindi/Marathi/Gujarati query find the right English precedents?

**Statutory relevance** = share of the top-5 retrieved cases that cite a section (or act) of the query. 360 queries, English oracle anchors as reference. Chance level: **0.014** (section), **0.063** (act). Source: `results/stage10_statutory_relevance_embedding_only.csv`.

| Section-level p@5 | en | hi | mr | gu |
|---|---|---|---|---|
| bge_m3 | 0.275 | 0.237 | 0.250 | 0.244 |
| e5_large_instruct | 0.315 | 0.261 | 0.225 | 0.206 |
| e5_base | 0.302 | 0.254 | 0.226 | 0.241 |
| labse | 0.269 | 0.260 | 0.241 | 0.254 |

| Act-level p@5 | en | hi | mr | gu |
|---|---|---|---|---|
| bge_m3 | 0.715 | 0.601 | 0.620 | 0.597 |
| e5_large_instruct | 0.694 | 0.517 | 0.454 | 0.448 |
| e5_base | 0.697 | 0.601 | 0.546 | 0.489 |
| labse | 0.571 | 0.536 | 0.532 | 0.529 |

**Reading it**

- Retrieval works: section-level precision is **15 to 22 times chance** and act-level 7 to 11 times chance, in every language and encoder.
- The **cost of crossing languages is visible at act level**: bge_m3 falls from 0.715 (English) to 0.60-0.62, e5_base from 0.697 to 0.49-0.60, e5_large_instruct from 0.694 to 0.45-0.52. LaBSE degrades least but starts lowest.
- **bge_m3 is best or tied-best at act level in all three Indian languages.** At section level the encoders are statistically indistinguishable (intervals overlap, ±5 points).
- e5_large_instruct is the weakest cross-lingually despite being the largest model. The query instruction was applied as intended (instruction on the query side only, see `src/embed.py`), so a missing prefix is not the explanation; the cause was not determined. The instruction is written in English, which may matter for Indic queries.
- bge_m3 was chosen as the headline encoder after seeing these 360 queries. This is a retrieval metric rather than the verdict metric, so the selection risk is small, but it is disclosed.

### 6.2 Verdict prediction by embedding retrieval alone (kNN vote, k = 11)

Accuracy of the majority outcome among the 11 nearest labelled English cases (360 queries, baseline 0.503, 95% CI about ±5 points):

| | en | hi | mr | gu |
|---|---|---|---|---|
| bge_m3 | 0.556 | 0.578 | 0.561 | 0.572 |
| e5_large_instruct | 0.569 | 0.531 | 0.522 | 0.511 |
| e5_base | 0.553 | 0.550 | 0.503 | 0.572 |
| labse | 0.583 | 0.519 | 0.550 | 0.525 |

bge_m3 is slightly above the baseline in all four languages (0.556-0.578; the lower interval bounds are only 0.505-0.525), and the other encoders range from 0.503 to 0.583, mostly within noise of the baseline. Neighbour overlap with the English query's neighbours is low (top-10 overlap 0.27-0.47), yet accuracy is similar, so overlap cannot serve as a quality measure here.

**Effect of the anchor rerank on retrieval (bge_m3, alpha = 0.6, `results/stage10_headline.csv`).** Among the 166 queries with a section anchor, the chance that the top-5 contain a case sharing a section with the query rises from 0.548 / 0.504 / 0.519 / 0.519 (cosine) to 0.763 / 0.741 / 0.726 / 0.711 (hybrid) for en / hi / mr / gu, about +0.19 to +0.24. kNN accuracy at k = 5 changes only a little (cosine 0.543-0.570, hybrid 0.590-0.617; only Marathi has an interval above 0, +0.073 [0.013, 0.130]).

### 6.3 End-to-end LLM verdicts (test set, 300 cases)

![Fig 3](results/figures/fig3_accuracy.png)

Accuracy at the 0.5 threshold; baseline 0.503. Source: `results/stage12_main.csv` (intervals are 95% bootstrap). Median split = accuracy after thresholding at the median probability (label-free).

| System | Lang | Acc @0.5 [95% CI] | Share ALLOWED | Macro-F1 @0.5 | Median-split acc | AUC [95% CI] |
|---|---|---|---|---|---|---|
| none | en | 0.500 [0.443, 0.557] | 0.903 | 0.405 | 0.483 | 0.503 [0.442, 0.573] |
| none | hi | 0.480 [0.427, 0.540] | 0.943 | 0.355 | 0.473 | 0.457 [0.395, 0.523] |
| none | mr | 0.490 [0.430, 0.550] | 0.953 | 0.361 | 0.497 | 0.484 [0.420, 0.552] |
| none | gu | 0.503 [0.447, 0.560] | 0.960 | 0.372 | 0.490 | 0.468 [0.403, 0.529] |
| cos | en | 0.537 [0.480, 0.593] | 0.767 | 0.502 | 0.563 | 0.599 [0.534, 0.665] |
| cos | hi | 0.547 [0.490, 0.607] | 0.803 | 0.502 | 0.570 | 0.601 [0.536, 0.662] |
| cos | mr | 0.520 [0.467, 0.577] | 0.857 | 0.452 | 0.543 | 0.569 [0.502, 0.631] |
| cos | gu | 0.537 [0.477, 0.593] | 0.860 | 0.469 | 0.587 | 0.617 [0.554, 0.678] |
| hyb | en | 0.537 [0.483, 0.600] | 0.760 | 0.504 | 0.570 | 0.609 [0.543, 0.678] |
| hyb | hi | 0.557 [0.500, 0.617] | 0.760 | 0.525 | 0.597 | 0.625 [0.557, 0.685] |
| hyb | mr | 0.533 [0.477, 0.593] | 0.823 | 0.480 | 0.583 | 0.629 [0.568, 0.692] |
| hyb | gu | 0.520 [0.463, 0.573] | 0.843 | 0.457 | 0.583 | 0.624 [0.559, 0.683] |

Note: macro-F1 here is recomputed from the verdict file and matches `stage12_main.csv`. The macro-F1 column in `stage11_summary.csv` (for example 0.463 for none/en) is a different quantity and should not be quoted as macro-F1.

### 6.4 Stage 12 evaluation (threshold-free metrics, paired tests, consistency)

Full tables: `results/stage12_main.csv`, `results/stage12_paired.csv`, `results/stage12_consistency.csv`. Figures 4-6 are in `results/figures/`.

![Fig 4](results/figures/fig4_paired_effects.png)
![Fig 5](results/figures/fig5_bias_and_auc.png)
![Fig 6](results/figures/fig6_consistency.png)

**Threshold-free metrics.** Probabilities are saturated near 1 (best dev thresholds 0.994-0.999), so accuracy at 0.5 is a poor metric.

- **AUC:** no retrieval 0.457-0.503 (all intervals contain 0.5); with retrieval 0.569-0.629, and all 8 intervals lie above 0.5 (the lowest lower bound is 0.502, cos/mr).
- **Median-split accuracy:** none 0.473-0.497; cos 0.543-0.587; hyb 0.570-0.597. Several intervals now exclude the 0.503 baseline (for example hyb/hi 0.597 [0.537, 0.650]), so a weak signal exists that the 0.5 threshold hides.
- **Paired change vs `none` (`results/stage12_paired.csv`):**
  - Accuracy @0.5, averaged over 4 languages: cos +0.042 [0.005, 0.078], hyb +0.043 [0.008, 0.081].
  - Hindi: hyb +0.077 [0.020, 0.130] (McNemar p = 0.013), cos +0.067 [0.013, 0.117] (p = 0.021). After Holm correction over 12 tests: p = 0.154 and 0.226, so **not significant**. No other language is significant at 0.5.
  - AUC change: +0.085 to +0.168; 7 of 8 intervals exclude 0 (cos/mr [-0.005, 0.170] does not).
  - Median-split change, averaged: cos +0.080 [0.028, 0.128], hyb +0.098 [0.044, 0.147].
- **hyb vs cos:** accuracy @0.5 +0.002 [-0.025, 0.028] averaged; -0.017 to +0.013 by language; AUC change +0.007 to +0.060 by language (only mr has an interval above 0: +0.060 [0.007, 0.112]); no consistent difference.
- **Cross-lingual consistency** (`results/stage12_consistency.csv`, same case in different languages):

| | none | cos | hyb |
|---|---|---|---|
| Spearman of P(ALLOWED), pairs | 0.12-0.42 | 0.44-0.66 | 0.48-0.61 |
| Kappa @ median split, pairs | 0.09-0.35 | 0.28-0.50 | 0.32-0.52 |
| All 4 languages agree @ median split | 0.320 | 0.457 | 0.447 |
| Kappa @0.5, pairs | -0.07 to 0.11 | 0.31-0.41 | 0.31-0.39 |

At 0.5, `none` shows 0.80 four-way agreement only because it answers ALLOWED for 90-96% of cases; this is not real consistency. Chance four-way agreement at a median split is about 0.125.

## 7. Insights

1. **Retrieval of relevant law works across languages; verdict prediction from these inputs is weak.** Retrieved precedents cite the query's statutes 7-22 times more often than chance, yet neither kNN voting nor the LLM reaches an accuracy at 0.5 significantly above the majority baseline. Threshold-free metrics show a weak signal (AUC 0.57-0.63). This agrees with the leakage analysis: facts, issues and statute carry little outcome signal (query-only TF-IDF about 0.59). Statutory relatedness is not outcome relatedness.
2. **Most of the retrieval "gain" for the LLM is bias correction, not legal reasoning.** Without precedents the model says ALLOWED for 90-96% of cases, so on a balanced test set it sits at chance. With precedents that share falls to 76-86%, which is still above the true 50%. Because the model's P(ALLOWED) is saturated near 1 (the best dev threshold was 0.994-0.999), accuracy at 0.5 is a poor metric, which is why AUC and a label-free median split were added.
3. **Retrieval helps over no retrieval, weakly.** The paired accuracy change is +0.017 to +0.077; averaged over languages it is about +0.04 with an interval just above 0. Only Hindi is individually significant at 0.5 (cos +0.067, hyb +0.077), and not after Holm correction (p = 0.15-0.23). AUC and median-split gains are clearer (AUC +0.085 to +0.168), but the absolute level is still low.
4. **Anchors add nothing measurable to the verdict (RQ2).** Hybrid minus cosine accuracy is between -0.017 and +0.013 across languages (average +0.002 [-0.025, 0.028]). The anchors raise statutory relevance by construction (section hit@5 up by about 0.2 for bge_m3), but that does not change the predicted outcome.
5. **No clear cross-lingual penalty in verdict accuracy (RQ3, part 1).** Hindi, Marathi and Gujarati are within about ±0.04 of English, with overlapping intervals, because there is no outcome signal for translation to lose. This does not show translation is harmless; the retrieval-level results (6.1) do show a cost at act level.
6. **Translation noise is concentrated in numbers.** Acts are recovered almost always, section numbers less so, and Marathi and Gujarati are worse than Hindi. The translator's changed numbers and dropped year ranges set a ceiling on anchor quality that no extractor can fix.
7. **Explanations are in the right script but not verified.** Script share is 1.000 (en), 1.000 (hi), 0.996 (gu) and 0.979 (mr) over 15 explanations per language. The sampled Marathi explanation is close to meaningless, the Gujarati one invents a notification number ("No.of 1981"), and the sampled Hindi verdict is wrong. Script share shows the language is right and says nothing about fluency or faithfulness.
8. **Sample size limits every claim.** With 300 test cases the 95% interval on accuracy is about ±5.6 points, so only effects larger than roughly 6 points could be detected.

## 8. Research questions: summary of answers

| RQ | Answer |
|---|---|
| **RQ1** Which retrieval strategy works best for a non-English query over an English corpus? | Implemented and compared: English baseline, cross-lingual embeddings, no retrieval. Cross-lingual embedding retrieval finds statutorily relevant precedents far above chance; bge_m3 is best or tied-best at act level; no retrieval strategy gives an accuracy at 0.5 significantly above the baseline, although AUC rises from about 0.48 to 0.57-0.63 with retrieved precedents. Same-language corpus and translate-test were **not implemented**. |
| **RQ2** Do anchors / base-rate priors improve prediction beyond retrieval? | Anchors improve statutory relevance of retrieved cases but not verdict accuracy (accuracy differences within ±0.02, intervals include 0; section hit@5 rises by about 0.2). Base-rate priors were not tested separately end to end. |
| **RQ3** Is the prediction consistent across languages? | Accuracy is similar across languages. For the same case, retrieval makes outputs more consistent: Spearman 0.12-0.42 (none) vs 0.44-0.66 (cos) and 0.48-0.61 (hyb); four-way agreement at a median split 0.32 vs 0.46 and 0.45 (see 6.4). |

## 9. Limitations

- **Labels are inferred** from ruling text; the meaning of 0/1/2 is not documented. Class 2 (partial) is excluded.
- **Small test set:** 300 cases; 166 with a section anchor. Section-level comparisons carry roughly ±7.5 points of uncertainty.
- **Machine translation only.** Native-speaker review sheets exist (`results/review_{hi,mr,gu}.csv`, 40 sentence pairs each), but a completed native review is not reported here. <!-- TODO: state review status -->
- **Explanations unverified** beyond script share; only 15 per language were generated.
- **Not implemented:** same-language-corpus retrieval, translate-test, and the oracle-versus-extracted-anchor ablation through the LLM.
- Weak anchors (`SEC:?:n`) are ignored by the hybrid reranker.
- **Alpha and encoder selection:** alpha was tuned per encoder on the 60 dev cases, and bge_m3 was chosen as the headline encoder after seeing the 360 queries (a retrieval metric, not the verdict metric).
- **One LLM** (Qwen2.5-7B-Instruct), **single run** with one split seed; embeddings stored in fp16.
- The corpus contains verdict-like sentences from lower-court history in about 12% of queries; the main setting keeps them (ablation with masking: TF-IDF 0.592 -> 0.586).
- Held-out cases are excluded from the corpus (97% of input cases would otherwise retrieve themselves).

## 10. Repository layout

```
src/
  sections.py        parse a summary into query_text / ratio / ruling / ok
  mask.py            outcome masker (ablation)
  anchors.py         English anchor extractor
  anchors_indic.py   Hindi/Marathi/Gujarati extractor + invariant form
  translate_it.py    InLegalTrans translation of queries
  llm.py             prompts, verdict from next-token probabilities, explanations
scripts/
  rebuild_corpus.py  rebuilds data/processed/corpus_en.jsonl from raw HF data + split_ids.json
data/processed/      queries_{en,hi,mr,gu}.jsonl, split_ids.json, anchors_queries_*.jsonl
results/             review sheets, stage8/10/11/12 CSVs, stage11_verdicts.jsonl, figures/
logs/                dated work logs
```

Git-ignored (too large or regenerable): `data/raw/`, `data/interim/`, `data/processed/corpus_en.jsonl`, `data/processed/anchors_corpus_en.jsonl`.

Corpus embeddings (4 encoders, fp16) are stored in a private Hugging Face dataset `Ritu29/nyaysetu-indic-emb`, not in this repository.

## 11. Reproducing

The work was run stage by stage in Kaggle notebooks (Internet on; GPU needed for the embedding and LLM stages).

1. Add two Kaggle secrets: a GitHub token (`GH_TOKEN`, contents read/write on this repo) and a Hugging Face token (`HF_TOKEN`, with access to the gated `ai4bharat/indictrans2-en-indic-1B` tokenizer repo). A write token is needed to upload embeddings.
2. Clone the repository and run `python scripts/rebuild_corpus.py` if `corpus_en.jsonl` is missing (downloads about 1.5 GB; expects 52,947 rows and 0 held-out ids in the corpus).
3. Run the anchor self-test and require `24 of 24 passed | READY` before any evaluation.
4. Stage 9: encode the corpus and queries (fixed 512-token input); e5 models need `query:` / `passage:` prefixes.
5. Stage 10-12: retrieval, hybrid rerank, LLM verdicts, explanations, evaluation and figures.

Translation-environment fixes (pinned `transformers==4.53.2`, IndicTransToolkit installed from GitHub, fp16, OOM-safe batching) are only needed to translate additional queries.

## 12. Acknowledgements and references

- NyayaRAG: arXiv 2508.00709; code `ShubhamKumarNigam/NyayaRAG`; data `L-NLProc/NyayaRAG` on Hugging Face. This project re-implements the setting rather than copying code.
- Translation: IndicTrans2 (AI4Bharat) and its legal fine-tune `law-ai/InLegalTrans-En2Indic-1B`.
- Encoders: BGE-M3, multilingual-e5 (large-instruct and base), LaBSE.
- Case text is derived from Indian Kanoon via the NyayaRAG dataset; check the licence before any redistribution.
