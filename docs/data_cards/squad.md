# Data Card — SQuAD (V1 single-hop fallback)

## Identity
- **Dataset:** SQuAD v1.1 (Rajpurkar et al.) via HuggingFace `rajpurkar/squad`
- **Role in this repo:** single-hop slot when Tevatron/wikipedia-nq (or TriviaQA) does not load. Occupies the same 40-train / 150-eval budget as NQ in `configs/default.yaml` (`eval_nq` / `train_nq` are “single-hop counts,” not NQ-only).
- **Committed ranking run:** `e8a4423` (2026-08-24) — 150 eval SQuAD + 150 Hotpot. **Superseded** as the headline snapshot by Tevatron NQ (`d456d26`; reward/\(Q_{\mathrm{cal}}\) rescored 2026-09-04). Keep this card for the fallback path. The SQuAD table below was **not** rescored.

## Motivation
Easy single-hop items stress-test cost-aware stopping. That only works if the gold passage is real article text, not `{question} The answer is {gold}`. SQuAD article contexts satisfy that constraint when DPR NQ is too heavy.

## Composition (this repo slice)
- Built by `scripts/prepare_data.py` → `src/data/wiki_passages.py` (`kind: squad`)
- Counts in `data/processed/slice_meta.json`: `nq_corpus: squad`, `nq_hf_dataset: rajpurkar/squad`, `n_nq_anchor: 0`, `n_nq_wiki: 16`
- Train: 40 questions from SQuAD `train`; eval: 150 from `validation`
- Shared BM25 index: 80,000 passages (2,086 Hotpot slice + **16** unique SQuAD article contexts + 77,898 unused-Hotpot distractors)

## Preprocessing
- Question + answer aliases + the **full article context** for that example
- Many questions share the same Wikipedia article, so unique SQuAD passages are few (16 on this slice)
- Unused Hotpot contexts fill the shared 80k pool
- Answer-anchor construction is rejected in code and by ranking preflight

## Splits
- Merged with Hotpot in the same processed files (`dataset: squad`)
- `aggregate_metrics` reports SQuAD under `by_dataset.squad`

## Known limitations
- **Not NQ.** Scope Memo V2 names Natural Questions. This is an honest fallback, not a substitute for DPR Wikipedia 100-word passages.
- Only 16 unique gold passages vs 77k Hotpot distractors. BM25 recall@5 on this slice is **0.633** (55/150 miss@5) — a real ranking split, but also a sparse-gold artifact.
- Short-span EM is strict: a prediction can merge two gold aliases and score EM = 0 with F1 ≈ 0.7 (see `docs/RESULTS.md` §4).
- Synthetic fallback is not SQuAD distribution

## Ranking snapshot (`e8a4423`, Qwen2.5-3B-Instruct)

| Policy | EM | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.400 | 60/150 | 16 | 1.69e-4 | 0.729 |
| rule_based | 0.420 | 63/150 | 15 | 5.03e-4 | 0.710 |
| max_tools | 0.400 | 60/150 | 17 | 7.50e-4 | 0.618 |

Verify (`rule_based`): 0 contradiction / 24 neutral / 126 support. After every verify the frozen policy stops.

## Ethical / license notes
- Follow SQuAD / HuggingFace dataset terms
