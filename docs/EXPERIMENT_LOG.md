# Experiment Log

Append-only notes while running pilots. Prefer short factual entries. **Each dated block names the run it belongs to.** Do not cite a number from this file without that run line.

For a full walkthrough of the current 80k ranking pilot (`2417c43`), see [`RESULTS.md`](RESULTS.md).

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

## 2026-08-24

**Run:** 80k-passage Qwen ranking pilot, commit `2417c43` (2026-08-23). Trajectories: `results/trajectories/rule_based_default.jsonl`.

- Trajectory check on `rule_based_default.jsonl` (lexical NLI, 150 Hotpot + 150 NQ).
- Verify labels: Hotpot **14 contradiction / 34 neutral / 102 support**; NQ **support 150/150**.
- After `verify`, next action was `stop` on 300/300 items, including all 14 Hotpot contradictions. Zero re-retrieve, zero rewrite after contradiction.
- Implication: the verifier discriminates on the hard split and is already in the env observation, but `rule_based` does not use it. That is a Milestone-3 policy gap, not a dead feature. Wrote up in `docs/RESULTS.md` §8 and `docs/IMPLEMENTATION_DECISIONS.md`.
