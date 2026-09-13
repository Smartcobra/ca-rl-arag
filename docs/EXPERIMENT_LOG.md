# Experiment Log

Append-only notes while running pilots. Prefer short factual entries. **Each dated block names the run it belongs to.** Do not cite a number from this file without that run line.

For a full walkthrough of the current 80k ranking (Tevatron NQ slice `d456d26`, **rescored 2026-09-04**) plus REINFORCE train (2026-09-09), 300-eval (2026-09-10, `default` learned = naive), and the 2026-09-13 free-cost sanity (`correctness_only` used the step cap; `lambda_zero` stayed retrieve→stop), see [`RESULTS.md`](RESULTS.md). The SQuAD fallback snapshot is `e8a4423`. The leaked-NQ 80k snapshot is `2417c43`.

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
- Implication: the verifier discriminates on the hard split and is already in the env observation, but `rule_based` does not use it. That is a Milestone-3 policy gap, not a dead feature. Wrote up in `docs/RESULTS.md` §9 and `docs/IMPLEMENTATION_DECISIONS.md`.

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
- Observation: single-hop is a ranking split (~40% EM), not a 146/150 ceiling. Reward still ranks naive > rule > max_tools. Full write-up at the time: `docs/RESULTS.md` (superseded 2026-08-27 by the Tevatron NQ table).

## 2026-08-27 (afternoon) — Tevatron NQ ranking pilot

**Run:** 80k-passage Qwen ranking pilot + reward ablation, commit `d456d26`. Tevatron/wikipedia-nq loaded (`datasets<3.0` + `trust_remote_code`). Slice: 150 Hotpot + 150 NQ. `slice_meta.json`: `nq_corpus: dpr_wikipedia_w100`, `nq_hf_dataset: Tevatron/wikipedia-nq`, `n_nq_anchor: 0`, `n_nq_wiki: 1450`, eval gold articles **847**, corpus 80,000 (2,086 Hotpot slice + 1,450 DPR golds + 1,509 DPR negatives + 74,955 unused-Hotpot distractors). BM25 R@5: Hotpot 0.927 (11 miss), NQ **0.587** (62 miss). Source: `results/metrics/pilot_summary_default.json`. Ablation JSON **was** regenerated on this slice (`reward_ablation_table.json`, stratified 100).

### HotpotQA (n=150)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.445 | 59/150 | 29 | 1.59e-4 | 0.654 |
| rule_based | 0.373 | 0.438 | 56/150 | 25 | 4.81e-4 | 0.590 |
| max_tools | 0.407 | 0.485 | 61/150 | 23 | 7.38e-4 | 0.587 |

### Natural Questions (n=150)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 0.348 | 41/150 | 16 | 1.84e-4 | 0.542 |
| rule_based | 0.273 | 0.352 | 41/150 | 15 | 5.27e-4 | 0.504 |
| max_tools | 0.293 | 0.358 | 44/150 | 18 | 7.55e-4 | 0.464 |

### Overall (mix-weighted)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | 0.598 |
| rule_based | 0.323 | 0.395 | 97/300 | 40 | 5.04e-4 | 4.0 | 0.547 |
| max_tools | 0.350 | 0.422 | 105/300 | 41 | 7.47e-4 | 7.0 | 0.526 |

- Action mix unchanged: naive 1 retrieve; rule retrieve+rerank+verify; max retrieve×3+rewrite+rerank+verify. Spend 1× / 2.9× / 4.3×. Latency 1089 / 1724 / 3387 ms.
- Hotpot `max_tools` vs naive: **3 recoveries / 1 regression**. Hotpot rule vs naive: **2 / 5**. NQ rule vs naive: **1 / 1** (tied). NQ max vs naive: **5 / 2** (net +3).
- Verify (`rule_based_default.jsonl`): Hotpot **14 contradiction / 31 neutral / 105 support**; NQ **0 / 18 / 132**. After every `verify`, next action is `stop` (300/300).
- Ablation (`rule_based`, stratified 100, EM 0.34): correctness_only 0.362 → grounding 0.547 → default 0.518 → high_cost_pressure 0.456. Grounding jump is real but smaller than leaked NQ.
- Observation: NQ on DPR Wikipedia is a ranking split (~27–29% EM), not a 146/150 ceiling. Reward still ranks naive > rule > max_tools. Full write-up at the time: `docs/RESULTS.md` (reward/\(Q_{\mathrm{cal}}\) superseded 2026-09-04 after the calibration fix; EM/F1/$ unchanged).

## 2026-09-04 — Calibration lazy-abstain fix; ranking snapshot rescored

**Run:** same 80k Tevatron-NQ Qwen ranking slice as `d456d26` (150 Hotpot + 150 NQ). Not a new GPU ranking job. `calibration_score` no longer treats gold-wrong as justified abstain (`src/rewards.py`). On-disk source: `results/metrics/pilot_summary_default.json` and `reward_ablation_table.json`.

### What changed vs 2026-08-27

EM/F1/$/n_correct/action mix are identical. Reward and \(Q_{\mathrm{cal}}\) dropped because lazy abstains (usable evidence) are now −0.2 instead of +0.6.

| Policy | overall reward (old → new) | overall Q_cal (old → new) |
|---|---|---|
| naive_rag | 0.598 → **0.580** | −0.017 → **−0.137** |
| rule_based | 0.547 → **0.531** | −0.040 → **−0.147** |
| max_tools | 0.526 → **0.509** | −0.018 → **−0.128** |

Hotpot reward: 0.631 / 0.570 / 0.569. NQ reward: 0.529 / 0.492 / 0.450. Ranking is still naive > rule > max_tools.

Ablation (`rule_based`, stratified 100, EM 0.34): presets without γ are unchanged (correctness_only 0.362, grounding 0.547, faithfulness_cost 0.519). Presets with calibration dropped: default 0.518 → **0.499**, lambda_zero 0.520 → **0.501**, high_cost_pressure 0.456 → **0.431**.

Latency on the on-disk JSON is 1100 / 1761 / 3443 ms (was 1089 / 1724 / 3387). Tokens and $ match the 08-27 table.

Observation: closing the tautology does not change the frozen-policy ranking. It does change the learning signal: always-abstain is no longer easy reward. Full write-up: `docs/RESULTS.md`. Scoring table: `docs/REWARD_DESIGN.md`.

## 2026-09-09 — First REINFORCE train curve (train split only)

**Run:** `python scripts/train_policy.py` on the locked 100-example train file (60 Hotpot + 40 NQ). Same 80k Tevatron-NQ corpus and 2026-09-04 calibration rule as the ranking snapshot. MLP `10 → 16 → 5` (261 params), `lr=0.003`, 5 epochs, `reward_preset=default`. Source: `results/metrics/train_policy_curve.json` (`"split": "train"`, `n=100`). **Not a GPU re-ranking of the 300.**

| Epoch | mean reward | mean EM | mean steps | mean retrieve | mean verify |
|---|---:|---:|---:|---:|---:|
| 1 | 0.649 | 0.42 | 4.75 | 1.62 | 0.46 |
| 2 | 0.620 | 0.39 | 4.20 | 1.39 | 0.46 |
| 3 | 0.577 | 0.36 | 4.45 | 1.41 | 0.55 |
| 4 | 0.564 | 0.36 | 4.30 | 1.44 | 0.50 |
| 5 | 0.643 | 0.41 | 4.01 | 1.30 | 0.39 |

- Best train reward: epoch 1 (0.649). Last epoch: 0.643. Curve is noisy, not monotonic.
- Did **not** collapse to retrieve→stop (naive is 2 steps / 0 verify). Did **not** explode to max-tools (7 steps / 3 retrieves).
- Verify stayed in 0.39–0.55 — the intern fact-checks on about half the train tickets (sampling). Eval argmax (2026-09-10) did not.
- Do not compare 0.649 to naive 0.580. Different questions (100 train vs 300 eval).

300-eval: see the 2026-09-10 block. Full write-up: `docs/RESULTS.md` §6.

## 2026-09-10 — Learned 300-eval (frozen argmax)

**Run:** `python scripts/run_pilot.py --policies learned` on the locked 300-eval (150 Hotpot + 150 NQ). Same 80k Tevatron-NQ corpus and 2026-09-04 calibration rule. Source: `results/metrics/learned_default.json`, `results/trajectories/learned_default.jsonl`. **Not a new train.** `--policies learned` rewrote `pilot_summary_default.json` to naive + learned only; rule / max remain in their own JSON files.

### Overall (n=300)

| Policy | EM | F1 | n_correct | steps | retrieve | verify | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.333 | 0.397 | 100/300 | 2.0 | 1.0 | 0.0 | 1.72e-4 | 0.580 |
| learned | 0.333 | 0.397 | 100/300 | 2.0 | 1.0 | **0.0** | 1.72e-4 | 0.580 |

Hotpot 59/150 both. NQ 41/150 both. Predictions identical **300/300**. Action mix: retrieve→stop on every learned row. `verify_out` is null.

- Train curve had verify 0.39–0.55 and ~4 steps because training **samples**. Eval **argmax** is the naive mode.
- Reward tie is not a win. The controller did not use the verify signal that `rule_based` also ignores after the fact-check.
- Full write-up: `docs/RESULTS.md` §6.

## 2026-09-13 — Free-cost trainer sanity (`correctness_only` vs `lambda_zero`)

**Run:** one Colab session, notebook `notebooks/FreeCost_Trainer_Sanity_CA_RL_ARAG.ipynb`. Same 80k Tevatron-NQ corpus, 100-train / 300-eval, 261-param MLP, 5 epochs. Separate checkpoints so `learned_policy.pt` / `train_policy_curve.json` stay the `default` run.

```text
python scripts/train_policy.py --reward-preset correctness_only \
  --checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --curve results/metrics/train_policy_curve_correctness_only.json
python scripts/run_pilot.py --reward-preset correctness_only \
  --policies naive_rag,learned \
  --learned-checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --no-figures
# then the same pair with --reward-preset lambda_zero and *_lambda_zero.pt / curve
```

### `correctness_only` train (`train_policy_curve_correctness_only.json`)

| Epoch | mean reward | mean EM | mean steps | mean retrieve | mean verify |
|---|---:|---:|---:|---:|---:|
| 1 | 0.450 | 0.42 | 4.66 | 1.62 | 0.50 |
| 2 | 0.383 | 0.36 | 4.14 | 1.55 | 0.41 |
| 3 | 0.403 | 0.38 | 4.26 | 1.61 | 0.53 |
| 4 | **0.464** | **0.43** | 5.49 | 1.80 | 1.06 |
| 5 | 0.402 | 0.37 | 5.84 | 1.84 | 1.20 |

Sampling moved toward more tools. Did not collapse to two steps.

### `correctness_only` 300-eval (`learned_correctness_only.json`)

| Policy | EM | n_correct | steps | retrieve | rewrite | verify | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.333 | 100/300 | 2.0 | 1.0 | 0.0 | 0.0 | 0.365 |
| learned | 0.340 | 102/300 | **8.0** | **3.0** | **2.0** | **2.0** | 0.378 |

Hotpot 60/150 vs naive 59 (2 recoveries / 1 regression). NQ 42/150 vs naive 41 (7 / 6). Overall 9 recoveries / 7 regressions. Predictions differ 38/300. **300/300 hit max_steps=8.** Dominant path (245/300): retrieve → rewrite ×2 → verify → retrieve ×2 → verify → stop.

### `lambda_zero` train + 300-eval

Train curve matches `default` homework (reward 0.651 / 0.621 / 0.579 / 0.566 / 0.648; steps 4.75 → 4.03; verify 0.39–0.55). Source: `train_policy_curve_lambda_zero.json`.

Eval (`learned_lambda_zero.json`): retrieve→stop **300/300**, verify 0, predictions identical to naive **300/300**, reward 0.581 = naive 0.581.

- **Trainer works.** When only \(Q_{\mathrm{ans}}\) pays, frozen argmax leaves two steps and uses retrieve/verify (here: the step cap).
- **`lambda_zero` still collapses.** Dollars/latency off is not enough; \(P_{\mathrm{act}}=0.02\) still makes extra tools −EV.
- Quality ceiling is still small: +2 EM vs naive, below `max_tools` 105/300 under `default`.
- Full write-up: `docs/RESULTS.md` §6 (Free-cost trainer sanity).
