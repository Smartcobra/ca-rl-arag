# Data Card — Natural Questions (V1 slice)

## Identity
- **Dataset:** Natural Questions via HuggingFace `Tevatron/wikipedia-nq` when `--hf` is used (DPR Wikipedia 100-word passages, Karpukhin et al. 2020)
- **Fallbacks:** `Tevatron/wikipedia-trivia`, `Tevatron/wikipedia-squad`, or `rajpurkar/squad` — all ship real passages. Never `nq_open` answer-anchors.
- **Role (Scope Memo V2):** single-hop train mix; teaches when *not* to over-retrieve
- **On-disk slice right now:** Tevatron NQ **did** load. `slice_meta.json` has `nq_corpus: dpr_wikipedia_w100` / `nq_hf_dataset: Tevatron/wikipedia-nq` / `n_nq_anchor: 0`. Ranking quality numbers: commit `d456d26`. Reward/\(Q_{\mathrm{cal}}\) on disk: 2026-09-04 rescore.

## Motivation
Easy single-hop items stress-test cost-aware stopping. Quality-only agents often over-retrieve; NQ exposes that failure mode. That only works if the gold passage is real Wikipedia, not the question plus `The answer is {gold}`.

## Composition (this repo slice)
- Built by `scripts/prepare_data.py` → `src/data/wiki_passages.py`
- Counts in `data/processed/slice_meta.json`: `n_nq_wiki: 1450`, `n_nq_wiki_neg: 1509`, eval `n_gold_articles: 847` (not 7)
- Train: 40 from Tevatron `train`; eval: 150 from Tevatron `dev`
- Shared BM25 index: 80,000 passages (2,086 Hotpot slice + 1,450 DPR golds + 1,509 DPR negatives + 74,955 unused-Hotpot distractors)

## Preprocessing
- Question + answer aliases + **positive Wikipedia passages** from Tevatron/DPR
- A few DPR negatives per query are indexed as non-gold Wikipedia distractors
- Unused Hotpot contexts still fill the shared 80k pool
- Answer-anchor construction (`{question} The answer is {gold}`) is rejected in code and by ranking preflight

## Splits
- Train from Tevatron `train`; eval from Tevatron `dev` (DPR NQ test family)
- Merged with Hotpot in the same processed files (`dataset` field discriminates)

## Known limitations
- We index gold DPR passages + capped negatives + Hotpot distractors, **not** the full 21M `wiki_dpr` dump (too heavy for Colab)
- Short-answer EM/F1 only (no long-answer)
- Synthetic fallback is not NQ distribution
- Historical RESULTS (`2417c43`) used leaked anchors (NQ 146/150). Those metrics are not comparable to this DPR NQ run
- Tiny-corpus max-tools write-up (`34e6585`, NQ 148/150): [`NQ_MAX_TOOLS_ANALYSIS.md`](../NQ_MAX_TOOLS_ANALYSIS.md)
- SQuAD fallback ranking (`e8a4423`) is a different corpus

## Ranking snapshot (`d456d26` slice, Qwen2.5-3B-Instruct; reward rescored 2026-09-04)

| Policy | EM | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 41/150 | 16 | 1.84e-4 | 0.529 |
| rule_based | 0.273 | 41/150 | 15 | 5.27e-4 | 0.492 |
| max_tools | 0.293 | 44/150 | 18 | 7.55e-4 | 0.450 |

BM25 R@5 **0.587** (62/150 miss@5). Verify (`rule_based`): 0 contradiction / 18 neutral / 132 support. After every verify the frozen policy stops.

## Ethical / license notes
- Follow Natural Questions / DPR / Tevatron / HuggingFace dataset terms
