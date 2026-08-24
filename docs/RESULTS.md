# Results Guide — What Was Produced and How to Read It

**Run this doc describes:** 80k-passage Qwen ranking pilot (300 eval: 150 Hotpot + 150 NQ). Metrics committed in `2417c43` (2026-08-23). Headline artifact: `results/metrics/pilot_summary_default.json`. Ablation: same corpus, stratified 100, `results/metrics/reward_ablation_table.json` (same commit). Section 6 (synthetic) is a different run (extractive, 2026-08-07). Section 8 (verifier labels) uses the 80k `rule_based_default.jsonl` from `2417c43`.

This document describes the **Milestone 2 pilot results**: where files live, what each metric means, how to interpret the current numbers, and known limitations.

For how to regenerate results, see [`HOW_TO_RUN.md`](HOW_TO_RUN.md).  
For append-only run notes, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).  
NQ max-tools mechanism (tiny-corpus 2,276-passage run `34e6585`, NQ 148/150): [`NQ_MAX_TOOLS_ANALYSIS.md`](NQ_MAX_TOOLS_ANALYSIS.md).

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
| Corpus | **80,000** passages (2,086 Hotpot slice + 190 NQ anchors + 77,724 unused-Hotpot distractors) |
| BM25 gold recall | Hotpot R@1 0.76 / R@5 **0.927** (11 miss@5); NQ R@5 **1.0** |
| Eval limit in pilot | Full eval (`limit: null`) |
| Reward preset | `default` (see `configs/reward_weights.yaml`) |
| Retriever | BM25 |
| Generator | **Qwen2.5-3B-Instruct** |
| Verifier | Lexical NLI |
| Policies | `naive_rag`, `rule_based`, `max_tools` |

These runs validate the **pipeline, costs, and frozen-policy reward ranking**, not SOTA Hotpot/NQ accuracy. NQ is still an answer-anchor ceiling (`The answer is {ans}` in `prepare_data.py`); the distractor pool does not fix that. Hotpot retrieval is no longer near-perfect.

Counts and recall: `data/processed/slice_meta.json`.

---

## 3. Main policy comparison (Qwen, 300 eval examples, 80k corpus)

**Comparison in one line:** Hotpot is 59 / 58 / 61. `max_tools` is +2 vs naive (3 recoveries / 1 regression) but ~4.6× $, so reward ranks **naive > rule > max_tools**. NQ is 146/150 for every policy.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`, `force_yes_no: true`, 80k-passage index).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.443 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.653** |
| rule_based | 0.387 | 0.446 | 58/150 | 28 | 0.187 | 4.81e-4 | 0.604 |
| max_tools | **0.407** | **0.476** | **61/150** | 23 | 0.153 | 7.37e-4 | 0.582 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.683 | 0.715 | 205/300 | 29 | 1.56e-4 | 2.0 | **1.076** |
| **rule_based** | 0.680 | 0.717 | 204/300 | 28 | 4.75e-4 | 4.0 | 1.031 |
| **max_tools** | 0.690 | 0.732 | 207/300 | 23 | 7.21e-4 | 7.0 | 0.989 |

### Natural Questions (n=150; saturated)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.973 | 0.988 | 146/150 | 0 | 1.53e-4 | 1.499 |
| rule_based | 0.973 | 0.988 | 146/150 | 0 | 4.69e-4 | 1.458 |
| max_tools | 0.973 | 0.988 | 146/150 | 0 | 7.06e-4 | 1.396 |

### How to read this table

**Retrieval**
- The 80k unused-Hotpot pool is why this run is not a redo of the 2,276-passage snapshot.
- Hotpot gold is missing from BM25 top-5 on **11/150** items (R@5 0.927). First retrieve can fail.
- NQ R@5 stays 1.0 because each item still has an answer-anchor passage.

**Quality**
- **Read Hotpot, not Overall.** NQ is 146/150 for every policy (anchor ceiling; 148→146 vs the tiny-corpus run).
- Hotpot is a few-hit race: 59 / 58 / 61. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). Too small to call a quality winner.
- Versus the 2,276-passage Qwen run: naive 60→59, rule 58→58, max 59→61. Abstain 20/17/19 → **29/28/23**. Harder search → more refusals, and a bit more room for extra retrieves.

**Cost / behavior**
- **naive_rag:** 1 retrieve → answer; 1026 ms; 693 tokens.
- **rule_based:** rerank + verify (4 steps; 3.0× $; 1600 ms; 1406 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.6× $; 3159 ms; 1774 tokens). High-cost reference. Latency is up vs the 2k index because each BM25 call scores 80k passages.

**Reward**
- Overall: naive 1.076 > rule 1.031 > max_tools 0.989. Hotpot: 0.653 > 0.604 > 0.582. Extra tools can move a couple of Hotpot answers; λ/$ still decides the ranking.

**$/correct**
- Overall $/correct is 2.28e-4 / 6.99e-4 / 1.05e-3.

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.699 | 0.834 | 0.175 | 0.030 |
| rule_based | 0.698 | 0.833 | 0.169 | 0.032 |
| max_tools | 0.711 | 0.851 | 0.160 | 0.031 |

- **Q_ans / Q_ground overall are NQ-inflated.** On Hotpot, Q_ans is 0.418 / 0.416 / 0.441; Q_ground is 0.668 / 0.667 / 0.702 (down vs the tiny-corpus run because gold titles compete with 77k distractors).
- **Q_cal is higher overall than the last 300-run** because Hotpot abstain rose to ~15–19%. That is retrieval hardness, not the old parser leak.

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

**Run this section describes:** same 80k Qwen corpus as the ranking pilot, commit `2417c43`. Source: `results/metrics/reward_ablation_table.json`.  
Fixed policy: **rule_based**, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.67; Hotpot 19/50, NQ 48/50); only the **scalar reward** changes.

| Preset | overall reward | Hotpot reward | NQ reward | What it tests |
|---|---:|---:|---:|---|
| correctness_only | 0.691 | 0.408 | 0.974 | Search-R1-like outcome only |
| correctness_grounding | 0.952 | 0.530 | 1.374 | Add grounding (β) |
| correctness_faithfulness_cost | 0.957 | 0.506 | 1.408 | Add cost + hall + act penalties |
| **default** | 0.985 | 0.522 | 1.448 | Full objective (α,β,γ,λ,μ,…) |
| lambda_zero | 0.987 | 0.524 | 1.450 | Remove $ / latency terms |
| high_cost_pressure | 0.966 | 0.469 | 1.462 | Larger λ/μ (cheaper operating point) |

### Interpretation

- EM/F1/$ stay flat across presets because the **policy is frozen**; ablations only change how we **score** trajectories.
- Grounding terms raise reward a lot when support overlap is high (`correctness_grounding`).
- Cost pressure (`high_cost_pressure`) lowers reward for the same spend — useful later when a learned policy can choose fewer tools.
- On this price card, absolute $ is tiny, so λ effects are modest until you scale prices or tool counts; the relative ordering still moves as designed.

Full weight definitions: [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

---

## 6. Synthetic pilot (sanity / offline)

**Run this section describes:** extractive generator, closed synthetic corpus, 16 eval examples, 2026-08-07 (`docs/EXPERIMENT_LOG.md`). Not the 80k Qwen ranking run.

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

## 8. Verifier signal vs frozen policy (RL implication)

This is a trajectory observation, not a code change. It matters for how we design the learned policy.

**What `verify` is, in one sentence.** After the agent has some passages and a draft answer, the lexical NLI verifier checks whether the evidence *agrees* with that answer. It returns one of three labels:

| Label | Meaning in plain English |
|---|---|
| `support` | The passages look consistent with the answer. |
| `neutral` | The passages do not clearly agree or disagree. |
| `contradiction` | The passages look like they *disagree* with the answer. |

That label is stored on each trajectory as `verify_out`. The Gym env already puts `support` and `contradiction` into the observation, so a learned policy *can* see them.

**What the current run actually returned** (`rule_based_default.jsonl`, 150 + 150):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA (hard ranking split) | **14** | **34** | **102** |
| Natural Questions (answer-anchor ceiling) | 0 | 0 | **150 / 150** |

Read this the junior way: on NQ the verifier is a yes-man. Every item is `support`, because the gold string is planted in the corpus. On Hotpot it is *not* a yes-man. About one in ten answers is flagged as a contradiction, and another ~one in five is only `neutral`. That mix is exactly where a verify signal is useful: it lights up on the hard split, and stays quiet on the easy one.

**What `rule_based` does with that signal: nothing that changes the search.** On this run, the next action after `verify` was `stop` on **300 / 300** items — including all 14 Hotpot contradictions. It never re-retrieved. It never rewrote the query. (There is a small “if support is very low, retrieve once more” branch in the frozen policy, but it did not fire here, and it does not look at the `contradiction` label anyway.)

So we have a gap:

1. The **signal exists** and is informative where it needs to be (Hotpot).
2. The **frozen policy does not use it** to recover (no extra retrieve / rewrite after contradiction).

That gap is precisely where a learned policy should win: see `contradiction` / low support in the state, then spend another retrieve or rewrite only when that looks worth the cost. The frozen baselines cannot show that behavior, so they are not a fair ceiling on what verify is worth.

---

## 9. Takeaways for the paper / next milestone

1. **Pipeline OK:** data → 80k BM25 index → agent actions → NLI verify → cost → multi-component reward → logs on the locked 300-example eval (150 Hotpot + 150 NQ). Preflight requires eval=300 and corpus ≥ 50k before Qwen loads.
2. **Reporting contract:** every table is overall + Hotpot + NQ. Overall is mix-weighted. NQ is still saturated (anchors) and cannot rank policies.
3. **The distractor pool did what it was for.** Hotpot R@5 is 0.927 (11 misses), not ~1.0. Abstain rose (20→29 naive). `max_tools` recovers 3 Hotpot items and loses 1 (61 vs 59). Reward still ranks naive because spend is 1× / 3.0× / 4.6×.
4. **Verify is informative on Hotpot and unused by `rule_based`.** Lexical NLI returns 14 contradiction / 34 neutral / 102 support on Hotpot, and support 150/150 on NQ. After every verify — including the 14 contradictions — the frozen policy just stops. That unused state feature is a Milestone-3 win condition for RL, not a reason to drop verify.
5. **Next:** Milestone 3 is cost-aware RL. Extra tools can now change a few answers; a learned controller must **select** when that is worth the cost, including when verify says the current evidence contradicts the draft. NQ still needs a real Wikipedia/DPR index before it is a ranking split.


---

## 10. How to regenerate and refresh this doc

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
