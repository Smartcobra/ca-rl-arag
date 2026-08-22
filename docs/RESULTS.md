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

**Comparison in one line:** on Hotpot, `max_tools` is the worst frozen policy (34/150 vs naive 37), not a quality ceiling. Extra tools cost ~4.7× and lose EM, so the reward ranks **naive > rule > max_tools**.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.247 | 0.298 | 37/150 | 0.493 | 1.56e-4 | 0.551 |
| rule_based | 0.233 | 0.288 | 35/150 | 0.507 | 4.73e-4 | 0.494 |
| max_tools | 0.227 | 0.285 | 34/150 | 0.493 | 7.12e-4 | 0.435 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | abstain | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.583 | 0.612 | 175/300 | 0.280 | 1.48e-4 | 2.0 | 0.989 |
| **rule_based** | 0.573 | 0.604 | 172/300 | 0.290 | 4.59e-4 | 4.0 | 0.936 |
| **max_tools** | 0.567 | 0.600 | 170/300 | 0.283 | 6.95e-4 | 7.0 | 0.872 |

### Natural Questions (n=150; saturated)

| Policy | mean EM | mean F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.920 | 0.925 | 138/150 | 0.067 | 1.40e-4 | 1.427 |
| rule_based | 0.913 | 0.919 | 137/150 | 0.073 | 4.45e-4 | 1.377 |
| max_tools | 0.907 | 0.915 | 136/150 | 0.073 | 6.77e-4 | 1.310 |

### How to read this table

**Quality**
- **Read Hotpot, not Overall.** NQ is near-ceiling and almost flat (136–138 / 150).
- **`max_tools` underperforms on Hotpot.** 34/150 vs naive 37 and rule 35. Vs naive that is 4 regressions / 1 recovery. All four losses had evidence reordered or swapped after unconditional rewrite + 3rd retrieve + rerank-on-rewritten-query. When the top-5 stayed identical to naive (74/150), EM matched naive (0.270). The 2nd retrieve is a no-op (same query, same hits on 150/150). Rewrite changed 147/150 Hotpot queries and only once recovered a miss.
- NQ EM ≈ 0.91–0.92 is an answer-anchor ceiling, not a model success.

**Cost / behavior**
- **naive_rag:** cheapest and shortest (1 retrieve → answer; 496 ms; 640 tokens).
- **rule_based:** adds rerank + verify (4 steps; 3.1× mean $; 1032 ms; 1300 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.7× mean $; 1819 ms; 1665 tokens). This is a high-cost reference, not a better agent.

**Reward**
- Overall reward ranks naive 0.989 > rule 0.936 > max_tools 0.872 because Hotpot EM is flat-to-worse and extra tools cost money.
- Same order on Hotpot: 0.551 > 0.494 > 0.435. That is the λ/$ signal working as designed on frozen scripts, not a finding that agentic RAG cannot help after RL.

**$/correct**
- Prefer **mean $**, **per-dataset EM/F1**, and **reward** together. Overall $/correct is 2.54e-4 / 8.01e-4 / 1.23e-3.

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.598 | 0.784 | 0.288 | 0.003 |
| rule_based | 0.588 | 0.778 | 0.291 | 0.003 |
| max_tools | 0.583 | 0.790 | 0.280 | 0.003 |

- **Q_ans / Q_ground overall are NQ-inflated.** On Hotpot, Q_ans is 0.273 / 0.261 / 0.256.
- **max_tools has the highest Q_ground** (0.790 vs 0.784 / 0.778) from three retrieves, but it does not convert to extra EM.
- **Q_cal is positive on average** because Hotpot abstain is ~49–51% for all three policies. That is a parser/policy mix, not a calibration win until the ABSTAIN leak is fixed.

---

## 4. Example failure (why EM = 0 is often not a bug)

From trajectory logs, example `hotpot_eval_5adbf0a255429947ff17385a`:

| Field | Value |
|---|---|
| Question | Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood? |
| Gold | `no` |
| Prediction | `Esma Sultan Mansion` |
| EM / F1 | 0 / 0 |

Retrieval found the right pages; the extractive answerer returned a **title** instead of answering **yes/no**. That is a generator limitation, not a broken metric or env.

When you see `env_rollouts.jsonl` rows like:

```json
{"id": "...", "reward": 0.28, "em": 0.0, "f1": 0.0}
```

it means: that Gym episode ended with a wrong answer and a low reward. Rows with `em: 1.0` / high reward are successful episodes.

---

## 5. Reward-weight ablation results

Source: `results/metrics/reward_ablation_table.json`  
Fixed policy: **rule_based**, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.56); only the **scalar reward** changes.

| Preset | overall reward | Hotpot reward | NQ reward | What it tests |
|---|---:|---:|---:|---|
| correctness_only | 0.567 | 0.213 | 0.920 | Search-R1-like outcome only |
| correctness_grounding | 0.868 | 0.427 | 1.308 | Add grounding (β) |
| correctness_faithfulness_cost | 0.862 | 0.386 | 1.338 | Add cost + hall + act penalties |
| **default** | 0.912 | 0.440 | 1.383 | Full objective (α,β,γ,λ,μ,…) |
| lambda_zero | 0.913 | 0.441 | 1.385 | Remove $ / latency terms |
| high_cost_pressure | 0.889 | 0.383 | 1.396 | Larger λ/μ (cheaper operating point) |

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
3. **`max_tools` is a quality regression on Hotpot**, not a ceiling: 34/150 vs naive 37 (4 losses / 1 win). Unconditional rewrite/rerank scramble evidence; the 2nd retrieve never changes hits. Reward ranks naive because spend is 1× / 3.1× / 4.7× and EM is flat-to-worse. ABSTAIN parsing still leaks mixed strings (Hotpot abstain ~50%).
4. **Next:** Milestone 3 is cost-aware RL. A learned controller must **select** tools — always-on `max_tools` is the wrong Hotpot operating point. Parser leak still needs fixing before treating abstain/Q_cal as a result.


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
