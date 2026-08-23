# Results Guide — What Was Produced and How to Read It

This document describes the **Milestone 2 pilot results**: where files live, what each metric means, how to interpret the current numbers, and known limitations.

For how to regenerate results, see [`HOW_TO_RUN.md`](HOW_TO_RUN.md).  
For append-only run notes, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

---

## 1. Where results live

```text
results/
├── metrics/                          # Aggregated summaries (committed)
│   ├── pilot_summary_default.json    # All policies, one reward preset
│   ├── baseline_default.json         # Naive RAG only
│   ├── rule_based_default.json
│   ├── max_tools_default.json
│   ├── reward_ablation_table.json    # Compact reward-weight sweep (nests by_dataset)
│   ├── reward_ablation_by_dataset.json
│   └── ablation_rule_based_*.json    # Per-preset full summaries
├── figs/                             # Plots from metrics (via plot_results.py)
│   ├── policy_quality_reward.png     # overall, mix-weighted
│   ├── policy_cost.png
│   ├── policy_action_mix.png
│   ├── policy_reward_components.png
│   ├── policy_pareto_em_usd.png
│   ├── policy_by_dataset.png         # Hotpot vs NQ vs overall
│   ├── reward_ablation.png
│   └── reward_ablation_by_dataset.png
└── trajectories/                     # Per-example logs (JSONL; often gitignored)
    ├── baseline_default.jsonl
    ├── rule_based_default.jsonl
    ├── max_tools_default.jsonl
    └── env_rollouts.jsonl            # Short Gym env check dumps
```

| File type | Granularity | Typical use |
|---|---|---|
| `pilot_summary_*.json` | Overall + `by_dataset` (Hotpot / NQ) per policy | Comparison table; never cite overall alone |
| `*_default.json` | One policy summary | Drill into one system |
| `*.jsonl` trajectories | One row per question | Failure analysis (wrong EM, action loops, costs) |
| `env_rollouts.jsonl` | Tiny env sanity rows (`id`, `reward`, `em`, `f1`) | Confirms Gymnasium env scores episodes |
| `reward_ablation_table.json` | Same policy, different reward presets | Justify α/β/γ/λ choices |

---

## 2. Setup used for the main HuggingFace pilot

| Item | Value |
|---|---|
| Data | HotpotQA distractor + NQ Open (`prepare_data.py --hf`) |
| Slice | 100 train / **300 eval** locked (150 Hotpot + 150 NQ) |
| Eval limit in pilot | Full eval (`limit: null`) |
| Reward preset | `default` (see `configs/reward_weights.yaml`) |
| Retriever | BM25 |
| Generator | **Qwen2.5-3B-Instruct** |
| Verifier | Lexical NLI |
| Policies | `naive_rag`, `rule_based`, `max_tools` |

These runs validate the **pipeline, costs, and frozen-policy reward ranking**, not SOTA Hotpot/NQ accuracy. NQ is an answer-anchor ceiling (`The answer is {ans}` in `prepare_data.py`).

---

## 3. Main policy comparison (Qwen, 300 eval examples)

**Comparison in one line:** Hotpot is nearly tied (60 / 58 / 59). Extra tools still do not beat naive. Reward ranks **naive > rule > max_tools** on cost (~1× / 3.1× / 4.6×). NQ is 148/150 for every policy.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`, `force_yes_no: true`).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.400 | 0.488 | 60/150 | 20 | 0.133 | 1.60e-4 | 0.691 |
| rule_based | 0.387 | 0.481 | 58/150 | 17 | 0.113 | 4.81e-4 | 0.628 |
| max_tools | 0.393 | 0.494 | 59/150 | 19 | 0.127 | 7.20e-4 | 0.593 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.693 | 0.743 | 208/300 | 20 | 1.52e-4 | 2.0 | 1.102 |
| **rule_based** | 0.687 | 0.740 | 206/300 | 17 | 4.67e-4 | 4.0 | 1.050 |
| **max_tools** | 0.690 | 0.746 | 207/300 | 19 | 7.03e-4 | 7.0 | 1.002 |

### Natural Questions (n=150; saturated)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.987 | 0.998 | 148/150 | 0 | 1.45e-4 | 1.514 |
| rule_based | 0.987 | 0.998 | 148/150 | 0 | 4.54e-4 | 1.472 |
| max_tools | 0.987 | 0.998 | 148/150 | 0 | 6.86e-4 | 1.411 |

### How to read this table

**Quality**
- **Read Hotpot, not Overall.** NQ is 148/150 for every policy (answer-anchor ceiling).
- Hotpot is a 1–2 hit race: 60 / 58 / 59. `max_tools` vs naive is 2 regressions / 1 recovery.
- Versus the hedge-heavy Qwen 300-run: naive Hotpot 37 → 60, abstain 74 → 20. Versus the last run: NQ 147 → 148 (`is there a name for the at symbol` → `commercial at` after the yes/no detector tighten). Hotpot unchanged.

**Cost / behavior**
- **naive_rag:** 1 retrieve → answer; 502 ms; 666 tokens.
- **rule_based:** rerank + verify (4 steps; 3.1× $; 1083 ms; 1352 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.6× $; 1856 ms; 1717 tokens). High-cost reference, not a better agent.

**Reward**
- Overall: naive 1.102 > rule 1.050 > max_tools 1.002. Hotpot: 0.691 > 0.628 > 0.593. Quality is flat; λ/$ decides the ranking.

**$/correct**
- Overall $/correct is 2.19e-4 / 6.80e-4 / 1.02e-3.

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.718 | 0.860 | 0.152 | 0.031 |
| rule_based | 0.713 | 0.860 | 0.137 | 0.034 |
| max_tools | 0.718 | 0.869 | 0.146 | 0.031 |

- **Q_ans / Q_ground overall are NQ-inflated.** On Hotpot, Q_ans is 0.444 / 0.434 / 0.444.
- **Q_cal dropped** vs the hedge-heavy run because Hotpot abstain is ~12–13%, not ~50%. That is the intended effect of forcing answers.

---

## 4. Example failure (why EM = 0 is often not a bug)

From trajectory logs, example `hotpot_eval_5adbf0a255429947ff17385a`:

| Field | Value |
|---|---|
| Question | Were Scott Derrickson and Ed Wood of the same nationality? |
| Gold | `yes` |
| Prediction | `no` |
| EM / F1 | 0 / 0 |

Forced yes/no stopped the abstain (old run: `ABSTAIN`). The model now answers, but is biased to `no` (13/14 yes-no predictions). That is a generator error, not a broken metric.

When you see `env_rollouts.jsonl` rows like:

```json
{"id": "...", "reward": 0.28, "em": 0.0, "f1": 0.0}
```

it means: that Gym episode ended with a wrong answer and a low reward. Rows with `em: 1.0` / high reward are successful episodes.

---

## 5. Reward-weight ablation results

Source: `results/metrics/reward_ablation_table.json`  
Fixed policy: **rule_based**, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.69); only the **scalar reward** changes.

| Preset | overall reward | Hotpot reward | NQ reward | What it tests |
|---|---:|---:|---:|---|
| correctness_only | 0.715 | 0.442 | 0.989 | Search-R1-like outcome only |
| correctness_grounding | 0.991 | 0.593 | 1.389 | Add grounding (β) |
| correctness_faithfulness_cost | 0.997 | 0.571 | 1.424 | Add cost + hall + act penalties |
| **default** | 1.020 | 0.574 | 1.467 | Full objective (α,β,γ,λ,μ,…) |
| lambda_zero | 1.022 | 0.575 | 1.469 | Remove $ / latency terms |
| high_cost_pressure | 1.002 | 0.520 | 1.484 | Larger λ/μ (cheaper operating point) |

### Interpretation

- EM/F1/$ stay flat across presets because the **policy is frozen**; ablations only change how we **score** trajectories.
- Grounding terms raise reward a lot when support overlap is high (`correctness_grounding`).
- Cost pressure (`high_cost_pressure`) lowers reward for the same spend — useful later when a learned policy can choose fewer tools.
- On this price card, absolute $ is tiny, so λ effects are modest until you scale prices or tool counts; the relative ordering still moves as designed.

Full weight definitions: [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

---

## 6. Synthetic pilot (sanity / offline)

When quality is saturated (extractive EM ≈ 1.0 on a closed fact corpus), reward ranks by cost:

| Policy | mean EM | mean $ | mean reward |
|---|---:|---:|---:|
| naive_rag | 1.00 | 8.3e-5 | 1.525 |
| rule_based | 1.00 | 3.5e-4 | 1.470 |
| max_tools | 1.00 | 4.3e-4 | 1.423 |

This is the cleanest proof that the **cost-aware reward is wired correctly** before you introduce LLM noise.

---

## 7. What each trajectory JSONL field means

Each line in `results/trajectories/*_default.jsonl` is one question. Important fields:

| Field | Meaning |
|---|---|
| `dataset` | `hotpot_qa` or `natural_questions` — required for per-dataset aggregation |
| `em`, `f1` | Quality vs gold |
| `reward`, `q_ans`, `q_ground`, `q_cal`, `p_hall`, `c_tok`, `c_ret`, `c_lat`, `p_act`, `p_bud` | Reward breakdown |
| `total_usd`, `total_tokens`, `total_latency_ms` | Measured cost |
| `n_retrieve`, `n_rewrite`, `n_rerank`, `n_verify`, `n_steps` | Action mix |
| `trajectory` | Step-by-step actions and tool outputs |
| `retrieved` | Passage ids/titles/scores used at the end |
| `verify_out` | NLI support / contradiction / uncertainty (if verify ran) |
| `policy` | Which controller produced this run |

Use these rows for qualitative analysis (unnecessary retrieves, bad rewrites, yes/no failures, etc.).

---

## 8. Takeaways for the paper / next milestone

1. **Pipeline OK:** data → retrieve → agent actions → NLI verify → cost → multi-component reward → logs all run on the locked 300-example eval (150 Hotpot + 150 NQ).
2. **Reporting contract:** every table is overall + Hotpot + NQ. Overall is mix-weighted. NQ is saturated and cannot rank policies.
3. **Abstain fix moved the bottleneck.** Hotpot naive 37 → 60, abstain 74 → 20. Frozen tools are now a 1–2 hit race (60 / 58 / 59). Reward still ranks naive because spend is 1× / 3.1× / 4.6×.
4. **Next:** Milestone 3 is cost-aware RL. A learned controller must **select** tools. Yes/no still biases to `no` (9/14). Remaining Hotpot abstain is ~12–13%.


---

## 9. How to regenerate and refresh this doc

```bash
python scripts/prepare_data.py --hf
python scripts/run_pilot.py --run-env-check
python scripts/run_reward_ablation.py
python scripts/plot_results.py
```

Then update numbers in this file and in `EXPERIMENT_LOG.md` from:

- `results/metrics/pilot_summary_default.json` (overall + `by_dataset`)
- `results/metrics/reward_ablation_table.json` and `reward_ablation_by_dataset.json`

Figures land in `results/figs/` (overall bars plus `policy_by_dataset.png`).
