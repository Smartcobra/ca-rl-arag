# Implementation Decisions Log

Living document for methodology / discussion sections. Update as experiments proceed.

## 2026-08-07 — Milestone 2 bootstrap

### Verifier (locked)

- **Choice:** NLI-style verification (`lexical_nli` default).
- **Not chosen:** LLM-as-judge verifier (deferred; would add variance and couple verify cost to generator pricing).
- **Optional upgrade path:** `verification.backend: neural_nli` with `cross-encoder/nli-deberta-v3-base`, same `VerifyResult` schema.
- **Why now:** Reviewer asked to define verification early so experiments stay consistent.

### Generator

- **Choice for quality pilot:** local **HuggingFace** instruct model `Qwen/Qwen2.5-3B-Instruct` (`generation.backend: huggingface`). Weights cache under `~/.cache/huggingface`; inference on CUDA/MPS/CPU.
- **Offline / smoke path:** deterministic **extractive** generator (`configs/extractive.yaml`, and `smoke_test.py` forces extractive).
- **Not wired:** `openai` backend (raises clearly if selected).
- **Why:** Reviewer feedback — extractive pilots understate tool value; same agent/env/reward APIs, swap answerer only.

### Retriever

- **Choice:** BM25 over a shared passage corpus (`rank_bm25`).
- **Rerank:** lexical overlap reranker with the same interface as a future cross-encoder.
- **Deferred (V1.1):** `retrieve_semantic` / `retrieve_keyword` / `expand` (GRASP granularity).

### Datasets

- **Primary:** Natural Questions + HotpotQA (Scope Memo V2).
- **Pilot:** `scripts/prepare_data.py` builds a small slice; `--synthetic` enables offline debugging; `--hf` pulls HuggingFace `nq_open` + `hotpot_qa/distractor`.
- **NQ corpus note:** without a full Wikipedia dump in M2, HF mode adds answer-anchor passages for NQ items so the shared index remains solvable; Hotpot uses distractor contexts. Documented in data cards.

### Policies before RL

- `naive_rag`: retrieve → stop  
- `rule_based`: threshold policy (stable baseline)  
- `max_tools`: cost upper bound  
- `random`: exploration reference  
- **RL algorithms (GRPO/PPO) intentionally not optimized yet** — environment is ready (`src/rag_env.py`).

### Reward

- See `docs/REWARD_DESIGN.md`. Defaults justified; ablations encoded as named presets.

### Logging

- Trajectories: `results/trajectories/*.jsonl`  
- Metrics: `results/metrics/*.json`  
- Each episode stores action history, costs, reward components, EM/F1.
- Eval summaries **must** emit `by_dataset` (`hotpot_qa`, `natural_questions`) plus overall. Overall is mix-weighted and is not a ranking.

## 2026-08-20 — Per-dataset eval contract

- Trajectory rows already carry `"dataset"`. Aggregation now groups by that field and runs the same means/counts twice.
- Extra counts: `n_examples`, `n_correct`, `n_abstained`, `abstain_rate`.
- Console, `pilot_summary_*.json`, ablation tables, and plots all show Hotpot and NQ separately. Overall stays as a mix-weighted headline after the mix is visible.
- This does **not** make Hotpot gaps significant. NQ is saturated (answer-anchor passages). Ranking waits on the locked 300-example eval (150+150), Hotpot-only.

## 2026-08-21 — Eval grown to 300 (balanced)

- Config lock: `eval_hotpot: 150`, `eval_nq: 150` (train stays 60+40). Balanced mix for the public table; always read Hotpot as the ranking split.
- `python scripts/prepare_data.py --hf` rebuilds `eval_slice.jsonl` / corpus. Corpus grows with more Hotpot distractor contexts. NQ still gets answer-anchor passages — that ceiling is not a model success.
- Pilot default is the **full eval file** (no prefix `--limit`). A debug `--limit` is stratified: `round(limit * n_ds / n_total)` per dataset, leftover rounding to hit `limit`. `--limit 40` on 150+150 is ~20+20, never 40 Hotpot + 0 NQ.
- Ablation default is a **stratified 100** from the same 300, labeled as a subset. Not the ranking table.
- Existing Qwen 40-ex metrics stay in `results/` until the 300-example GPU run. Do not rank from them.

## 2026-08-24 — Lexical NLI is informative; `rule_based` does not act on it

Trajectory counts from `results/trajectories/rule_based_default.jsonl` (150 Hotpot + 150 NQ):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA | 14 | 34 | 102 |
| Natural Questions | 0 | 0 | 150 |

Junior reading: `verify` is a “does the evidence agree with this answer?” check. On NQ it always says yes (answer-anchor ceiling). On Hotpot it actually mixes — 14 contradictions, 34 neutrals — so the signal discriminates on the hard split. That makes support / contradiction a real **state feature** for the policy, not dead logging.

`rule_based` still ignores it for search. After every `verify` on this run the next action was `stop` (300/300), including all 14 Hotpot contradictions. No re-retrieve, no rewrite. The frozen policy was built as a stable threshold script (retrieve → rerank → verify → stop on strong scores). It was not built to *react* to a contradiction.

Implication for RL: do not treat “verify did not help `rule_based`” as “verify is useless.” The env already exposes `verify_support` / `verify_contra`. A learned policy can condition retrieve / rewrite / stop on those values. That unused gap is a designed place for GRPO/PPO to beat the frozen baseline. Full write-up: `docs/RESULTS.md` §8.

## Observations template

| Date | Experiment | Observation | Implication |
|---|---|---|---|
| 2026-08-24 | Qwen 300-eval, lexical NLI, `rule_based` trajectories | Hotpot verify mix is 14 contradiction / 34 neutral / 102 support; NQ is support 150/150. After every verify, including all 14 contradictions, the next action is `stop`. | Verify is a useful state feature on the hard split. Frozen policy does not use it. Learned policy should. |
