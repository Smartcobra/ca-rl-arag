# Experiment Log

Append-only notes while running pilots. Prefer short factual entries. **Each dated block names the run it belongs to.** Do not cite a number from this file without that run line.

For a full walkthrough of the current 80k ranking pilot (`e8a4423`, Hotpot + SQuAD fallback), see [`RESULTS.md`](RESULTS.md). The earlier 80k leaked-NQ snapshot is `2417c43`.

## 2026-08-07

**Run:** synthetic extractive smoke/pilot (closed fact corpus, 16 eval examples). Not the HuggingFace ranking table.

- Created Milestone 2 codebase under `agentic_rag_rl/`.
- Locked verifier to NLI; reward defaults + ablation presets in `configs/reward_weights.yaml`.
- Smoke test OK (`scripts/smoke_test.py`).
- Synthetic pilot (16 eval examples, closed corpus): cost-aware ranking emerges even with EM=1.0 for all policies.

### Synthetic pilot results (`reward_preset=default`)

| Policy | mean EM | mean F1 | mean $ | mean retrieves | mean reward |
|---|---|---|---|---|---|
| naive_rag | 1.00 | 1.00 | 8.3e-5 | 1.00 | 1.525 |
| rule_based | 1.00 | 1.00 | 3.5e-4 | 1.38 | 1.470 |
| max_tools | 1.00 | 1.00 | 4.3e-4 | 3.00 | 1.423 |

Observation: with extractive answers saturated, the reward correctly ranks cheaper policies higher (naive > rule > max_tools). This is exactly the signal we want before RL — once NQ/Hotpot with a real Wikipedia index lowers EM, quality–cost trade-offs become non-trivial.

### Reward ablation (rule_based, same slice)

| Preset | mean reward | notes |
|---|---|---|
| correctness_only | 1.000 | = Q_ans |
| correctness_grounding | 1.400 | + β Q_ground |
| correctness_faithfulness_cost | 1.425 | + cost / hall / act |
| default | 1.470 | full objective |
| lambda_zero | 1.471 | no $ / latency term |
| high_cost_pressure | 1.471 | tiny absolute $ → weak λ effect on synthetic price card |

### HuggingFace pilot (NQ + Hotpot distractor slice)

**Run:** extractive generator, 40-example prefix slice, ~911 passages. Pipeline sanity only; do not compare to Qwen 300-eval.

- Prepared with `python scripts/prepare_data.py --hf`
- 100 train / 50 eval / 911 passages
- Eval pilot limit 40 (`default` reward preset)

| Policy | mean EM | mean F1 | mean $ | mean retrieves | mean reward | n_correct |
|---|---|---|---|---|---|---|
| naive_rag | 0.050 | 0.091 | 9.6e-5 | 1.00 | 0.361 | 2/40 |
| rule_based | 0.025 | 0.066 | 3.6e-4 | 1.00 | 0.285 | 1/40 |
| max_tools | 0.075 | 0.143 | 5.4e-4 | 3.00 | 0.303 | 3/40 |

Observation: extractive generator is a weak absolute QA backend on open Hotpot/NQ (expected). Max-tools spends more and recovers slightly higher F1; naive wins on reward because cost terms dominate when quality is low. Next quality jump = swap `generation.backend` to an instruct LLM while keeping the same env/reward.

## 2026-08-20

**Run:** re-summary of the existing Qwen 40-ex trajectories (no GPU rerun). Not the locked 300-eval.

- Per-dataset aggregation is now the eval contract: `aggregate_metrics` emits overall + `by_dataset.{hotpot_qa,natural_questions}` with `n_examples`, `n_correct`, `n_abstained`, `abstain_rate`.
- Re-summarized the existing Qwen 40-ex trajectories (no GPU rerun). Mix is 25 Hotpot + 15 NQ. NQ EM = 0.933 for all three policies; Hotpot EM = 0.160 / 0.120 / 0.200 (naive / rule / max_tools). Overall 0.45 / 0.425 / 0.475 is (Hotpot correct + 14 NQ correct) / 40.
- Not a ranking. Hotpot gaps are 1–2 hits. NQ cannot rank policies (answer-anchor ceiling). Wait for n=300.

## 2026-08-21

**Run:** config lock only (no new GPU numbers). Slice plan for the later 300-eval.

- Locked eval slice: 150 Hotpot + 150 NQ = 300 (`configs/default.yaml`). Train unchanged (60+40).
- Prefix `--limit` removed from the default path (it would now be 40 Hotpot + 0 NQ). Debug `--limit` is stratified.
- Ablation default: stratified 100 from this file, not a ranking table.
- Next GPU job: `python scripts/prepare_data.py --hf` then `python scripts/run_pilot.py --run-env-check` (full 300). Do not treat the 40-ex Qwen JSON as the 300 result.

## 2026-08-24 (morning) — leaked-NQ 80k verify check

**Run:** 80k-passage Qwen ranking pilot, commit `2417c43` (2026-08-23), **leaked NQ anchors**. Superseded later the same day by `e8a4423`. Trajectories at the time: `results/trajectories/rule_based_default.jsonl`.

- Trajectory check on `rule_based_default.jsonl` (lexical NLI, 150 Hotpot + 150 NQ).
- Verify labels: Hotpot **14 contradiction / 34 neutral / 102 support**; NQ **support 150/150**.
- After `verify`, next action was `stop` on 300/300 items, including all 14 Hotpot contradictions. Zero re-retrieve, zero rewrite after contradiction.
- Implication: the verifier discriminates on the hard split and is already in the env observation, but `rule_based` does not use it. That is a Milestone-3 policy gap, not a dead feature. Wrote up in `docs/RESULTS.md` §8 and `docs/IMPLEMENTATION_DECISIONS.md`.

## 2026-08-24 (afternoon) — SQuAD fallback ranking pilot

**Run:** 80k-passage Qwen ranking pilot, commit `e8a4423`. Tevatron/wikipedia-nq was too heavy; `--hf` fell back to `rajpurkar/squad`. Slice: 150 Hotpot + 150 SQuAD. `slice_meta.json`: `nq_corpus: squad`, `n_nq_anchor: 0`, `n_nq_wiki: 16`, corpus 80,000 (2,086 Hotpot slice + 16 SQuAD contexts + 77,898 unused-Hotpot distractors). BM25 R@5: Hotpot 0.927 (11 miss), SQuAD 0.633 (55 miss). Source: `results/metrics/pilot_summary_default.json`. Reward ablation JSON was **not** regenerated (still leaked-NQ `2417c43`).

### HotpotQA (n=150)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.443 | 59/150 | 29 | 1.59e-4 | 0.653 |
| rule_based | 0.387 | 0.446 | 58/150 | 28 | 4.81e-4 | 0.604 |
| max_tools | 0.407 | 0.476 | 61/150 | 24 | 7.37e-4 | 0.581 |

### SQuAD (n=150)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.400 | 0.466 | 60/150 | 16 | 1.69e-4 | 0.729 |
| rule_based | 0.420 | 0.468 | 63/150 | 15 | 5.03e-4 | 0.710 |
| max_tools | 0.400 | 0.451 | 60/150 | 17 | 7.50e-4 | 0.618 |

### Overall (mix-weighted)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.397 | 0.454 | 119/300 | 45 | 1.64e-4 | 2.0 | 0.691 |
| rule_based | 0.403 | 0.457 | 121/300 | 43 | 4.92e-4 | 4.0 | 0.657 |
| max_tools | 0.403 | 0.463 | 121/300 | 41 | 7.43e-4 | 7.0 | 0.600 |

- Action mix unchanged: naive 1 retrieve; rule retrieve+rerank+verify; max retrieve×3+rewrite+rerank+verify. Spend 1× / 3.0× / 4.5×. Latency 1037 / 1615 / 3265 ms.
- Hotpot `max_tools` vs naive: **3 recoveries / 1 regression**. SQuAD rule vs naive: **7 / 4**. SQuAD max vs naive: **8 / 8** (tied).
- Verify (`rule_based_default.jsonl`): Hotpot **14 contradiction / 34 neutral / 102 support**; SQuAD **0 / 24 / 126**. After every `verify`, next action is `stop` (300/300).
- Observation: single-hop is a ranking split (~40% EM), not a 146/150 ceiling. Reward still ranks naive > rule > max_tools. Full write-up: `docs/RESULTS.md`.
