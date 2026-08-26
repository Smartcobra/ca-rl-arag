# Results Guide — What Was Produced and How to Read It

**Run this doc describes:** 80k-passage Qwen ranking pilot (300 eval: 150 Hotpot + 150 **SQuAD**). Metrics committed in `e8a4423` (2026-08-24). Headline artifact: `results/metrics/pilot_summary_default.json`. Tevatron/wikipedia-nq was too heavy; `--hf` fell back to `rajpurkar/squad` article contexts (`slice_meta.json`: `nq_corpus: squad`, `n_nq_anchor: 0`). Section 5 (reward ablation) is a **different run** (leaked-NQ 80k, stratified 100, `2417c43`). Section 6 (synthetic) is extractive, 2026-08-07. Section 8 (verifier labels) uses the current `rule_based_default.jsonl` from `e8a4423`.

This document describes the **Milestone 2 pilot results**: where files live, what each metric means, how to interpret the current numbers, and known limitations.

For how to regenerate results, see [`HOW_TO_RUN.md`](HOW_TO_RUN.md).  
For append-only run notes, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).  
Historical leaked-NQ max-tools mechanism (tiny-corpus 2,276-passage run `34e6585`, NQ 148/150): [`NQ_MAX_TOOLS_ANALYSIS.md`](NQ_MAX_TOOLS_ANALYSIS.md). That write-up is **not** this ranking snapshot.

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
│   ├── policy_by_dataset.png         # Hotpot vs SQuAD vs overall
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
| `pilot_summary_*.json` | Overall + `by_dataset` (Hotpot / SQuAD) per policy | Comparison table; never cite overall alone |
| `*_default.json` | One policy summary | Drill into one system |
| `*.jsonl` trajectories | One row per question | Failure analysis (wrong EM, action loops, costs) |
| `env_rollouts.jsonl` | Tiny env sanity rows (`id`, `reward`, `em`, `f1`) | Confirms Gymnasium env scores episodes |
| `reward_ablation_table.json` | Same policy, different reward presets | Justify α/β/γ/λ choices — **stale mix**, see §5 |

---

## 2. Setup used for the main HuggingFace pilot

| Item | Value |
|---|---|
| Data | HotpotQA distractor + **SQuAD** (`prepare_data.py --hf`; Tevatron NQ fallback) |
| Slice | 100 train / **300 eval** locked (150 Hotpot + 150 SQuAD) |
| Corpus | **80,000** passages (2,086 Hotpot slice + **16** SQuAD article contexts + 77,898 unused-Hotpot distractors) |
| BM25 gold recall | Hotpot R@1 0.76 / R@5 **0.927** (11 miss@5); SQuAD R@1 0.447 / R@5 **0.633** (55 miss@5) |
| Eval limit in pilot | Full eval (`limit: null`) |
| Reward preset | `default` (see `configs/reward_weights.yaml`) |
| Retriever | BM25 |
| Generator | **Qwen2.5-3B-Instruct** |
| Verifier | Lexical NLI |
| Policies | `naive_rag`, `rule_based`, `max_tools` |

These runs validate the **pipeline, costs, and frozen-policy reward ranking**, not SOTA Hotpot/SQuAD accuracy. SQuAD golds are real article contexts (not `{question} The answer is {gold}`). Only 16 unique SQuAD passages sit in an 80k Hotpot-heavy index, so first-shot BM25 can fail. Hotpot retrieval is also no longer near-perfect.

Counts and recall: `data/processed/slice_meta.json`.

---

## 3. Main policy comparison (Qwen, 300 eval examples, 80k corpus)

**Comparison in one line:** Hotpot is 59 / 58 / 61. SQuAD is 60 / 63 / 60. Extra tools move a few answers on both splits, but spend is 1× / 3.0× / 4.5×, so reward ranks **naive > rule > max_tools**. Single-hop is no longer a saturated ceiling.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`, `force_yes_no: true`, 80k-passage index, SQuAD fallback).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.443 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.653** |
| rule_based | 0.387 | 0.446 | 58/150 | 28 | 0.187 | 4.81e-4 | 0.604 |
| max_tools | **0.407** | **0.476** | **61/150** | 24 | 0.160 | 7.37e-4 | 0.581 |

### SQuAD (n=150; ranking split — not saturated)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.400 | 0.466 | 60/150 | 16 | 0.107 | 1.69e-4 | **0.729** |
| rule_based | **0.420** | **0.468** | **63/150** | 15 | 0.100 | 5.03e-4 | 0.710 |
| max_tools | 0.400 | 0.451 | 60/150 | 17 | 0.113 | 7.50e-4 | 0.618 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.397 | 0.454 | 119/300 | 45 | 1.64e-4 | 2.0 | **0.691** |
| **rule_based** | 0.403 | 0.457 | 121/300 | 43 | 4.92e-4 | 4.0 | 0.657 |
| **max_tools** | 0.403 | 0.463 | 121/300 | 41 | 7.43e-4 | 7.0 | 0.600 |

### How to read this table

**Retrieval**
- The 80k unused-Hotpot pool is why first retrieve can fail.
- Hotpot gold is missing from BM25 top-5 on **11/150** items (R@5 0.927).
- SQuAD gold is missing from BM25 top-5 on **55/150** items (R@5 0.633). That is the distractor pool plus only 16 unique article contexts in the index — not answer-anchor leakage.

**Quality**
- **Read Hotpot and SQuAD, not Overall.** Both splits sit near 40% EM. The old leaked-NQ table (146/150 for every policy) is gone.
- Hotpot is still a few-hit race: 59 / 58 / 61. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). Too small to call a quality winner.
- SQuAD: `rule_based` is 63/150 vs 60/150 naive and max. Rule vs naive is **7 recoveries / 4 regressions** (net +3). Max vs naive is **8 / 8** (tied). Blind extra retrieves do not help SQuAD quality.
- Versus the leaked-NQ 80k run (`2417c43`): overall EM 0.68 → **0.40** because single-hop is no longer a copy-the-anchor ceiling. Hotpot is unchanged (59 / 58 / 61). Max abstain 23 → 24.

**Cost / behavior**
- **naive_rag:** 1 retrieve → answer; 1037 ms; 747 tokens.
- **rule_based:** rerank + verify (4 steps; 3.0× $; 1615 ms; 1518 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.5× $; 3265 ms; 1935 tokens). High-cost reference. Latency is up vs the 2k index because each BM25 call scores 80k passages.

**Reward**
- Overall: naive 0.691 > rule 0.657 > max_tools 0.600. Hotpot: 0.653 > 0.604 > 0.581. SQuAD: 0.729 > 0.710 > 0.618. Extra tools can move a couple of answers; λ/$ still decides the ranking.

**$/correct**
- Overall $/correct is 4.14e-4 / 1.22e-3 / 1.84e-3.

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.425 | 0.725 | 0.028 | 0.047 |
| rule_based | 0.430 | 0.723 | 0.026 | 0.044 |
| max_tools | 0.433 | 0.732 | 0.019 | 0.046 |

- **Q_ans / Q_ground are no longer NQ-inflated.** On Hotpot, Q_ans is 0.418 / 0.416 / 0.441; Q_ground is 0.668 / 0.667 / 0.697. On SQuAD, Q_ans is 0.433 / 0.444 / 0.426; Q_ground is 0.781 / 0.779 / 0.768.
- **Q_cal collapsed vs the leaked-NQ run** (overall ~0.17 → ~0.03) because single-hop is no longer always-correct. SQuAD Q_cal is slightly negative (−0.013 / −0.006 / −0.007): some abstains and confident wrongs on a hard index.

---

## 4. Example failures (why EM = 0 is often not a bug)

**Hotpot yes/no bias** — `hotpot_eval_5a8b57f25542995d1e6f1371` (also the first `env_rollouts.jsonl` row):

| Field | Value |
|---|---|
| Question | Were Scott Derrickson and Ed Wood of the same nationality? |
| Gold | `yes` |
| Prediction | `no` |
| EM / F1 | 0 / 0 |
| env reward | −0.52 |

Forced yes/no stopped a silent abstain. The model answers, but is biased to `no`. That is a generator error, not a broken metric.

**SQuAD span mismatch** — `squad_eval_56be4db0acb8001400a502ee`:

| Field | Value |
|---|---|
| Question | Where did Super Bowl 50 take place? |
| Gold aliases | `Santa Clara, California` / `Levi's Stadium` / longer venue string |
| Prediction | `Levi's Stadium in Santa Clara, California` |
| EM / F1 | 0 / 0.71 |

The span is right. Strict EM fails because the prediction is a merge of two gold aliases, not an exact string. Read F1 and trajectories before calling this a retrieval miss.

When you see `env_rollouts.jsonl` rows like:

```json
{"id": "...", "reward": 0.13, "em": 0.0, "f1": 0.0}
```

it means: that Gym episode ended with a wrong answer and a low reward. Rows with `em: 1.0` / high reward are successful episodes.

---

## 5. Reward-weight ablation results

**Run this section describes:** leaked-NQ 80k Qwen corpus, commit `2417c43` (2026-08-23). **Not** the current SQuAD ranking slice. Source: `results/metrics/reward_ablation_table.json` (file timestamps 2026-08-23; still keyed `natural_questions`). Re-run `python scripts/run_reward_ablation.py` on the SQuAD files before citing these as the same experiment.

Fixed policy: **rule_based**, **stratified 100** from the old 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.67; Hotpot 19/50, NQ 48/50); only the **scalar reward** changes.

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
- Grounding terms raise reward a lot when support overlap is high (`correctness_grounding`). On leaked NQ that overlap was artificial; expect a smaller jump after the SQuAD rebuild.
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
| `dataset` | `hotpot_qa` or `squad` on the current slice (`natural_questions` / `trivia_qa` if those sources load) — required for per-dataset aggregation |
| `em`, `f1` | Quality vs gold |
| `reward`, `q_ans`, `q_ground`, `q_cal`, `p_hall`, `c_tok`, `c_ret`, `c_lat`, `p_act`, `p_bud` | Reward breakdown |
| `total_usd`, `total_tokens`, `total_latency_ms` | Measured cost |
| `n_retrieve`, `n_rewrite`, `n_rerank`, `n_verify`, `n_steps` | Action mix |
| `trajectory` | Step-by-step actions and tool outputs |
| `retrieved` | Passage ids/titles/scores used at the end |
| `verify_out` | NLI support / contradiction / uncertainty (if verify ran) |
| `policy` | Which controller produced this run |

Use these rows for qualitative analysis (unnecessary retrieves, bad rewrites, yes/no failures, span mismatches, etc.).

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

**What the current run actually returned** (`rule_based_default.jsonl`, 150 + 150, commit `e8a4423`):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA (hard ranking split) | **14** | **34** | **102** |
| SQuAD (real article contexts) | **0** | **24** | **126** |

Read this the junior way: the verifier is **not** a yes-man on either split now. On leaked NQ it was support 150/150 because the gold string was planted. On SQuAD, 24 items are only `neutral` (mostly misses or abstains: 15/24 abstain, 2/24 EM-correct). On Hotpot it still mixes — about one in ten answers is flagged as a contradiction (8 of those 14 are nonetheless EM-correct), and another ~one in five is only `neutral` (28/34 abstain). That mix is exactly where a verify signal is useful.

**What `rule_based` does with that signal: nothing that changes the search.** On this run, the next action after `verify` was `stop` on **300 / 300** items — including all 14 Hotpot contradictions and all 24 SQuAD neutrals. It never re-retrieved. It never rewrote the query. (There is a small “if support is very low, retrieve once more” branch in the frozen policy, but it did not fire here, and it does not look at the `contradiction` label anyway.)

So we have a gap:

1. The **signal exists** and is informative on both splits (Hotpot contradictions; SQuAD neutrals).
2. The **frozen policy does not use it** to recover (no extra retrieve / rewrite after contradiction or neutral).

That gap is precisely where a learned policy should win: see `contradiction` / low support / `neutral` in the state, then spend another retrieve or rewrite only when that looks worth the cost. The frozen baselines cannot show that behavior, so they are not a fair ceiling on what verify is worth.

---

## 9. Takeaways for the paper / next milestone

1. **Pipeline OK:** data → 80k BM25 index → agent actions → NLI verify → cost → multi-component reward → logs on the locked 300-example eval (150 Hotpot + 150 SQuAD). Preflight requires eval=300 and corpus ≥ 50k before Qwen loads, and rejects leftover NQ answer-anchors.
2. **Reporting contract:** every table is overall + Hotpot + SQuAD. Overall is mix-weighted. Single-hop is now a ranking split (~40% EM), not a saturated ceiling.
3. **The distractor pool did what it was for.** Hotpot R@5 is 0.927 (11 misses). SQuAD R@5 is 0.633 (55 misses). Overall EM dropped 0.68 → 0.40 vs leaked NQ because copy-the-anchor is gone.
4. **Quality vs cost is now the right story on both splits.** Hotpot 59 / 58 / 61 (3 recoveries / 1 regression). SQuAD 60 / 63 / 60 (rule net +3; max tied 8/8). Reward still ranks naive because spend is 1× / 3.0× / 4.5×.
5. **Verify is informative and unused by `rule_based`.** Lexical NLI returns 14 contradiction / 34 neutral / 102 support on Hotpot, and 0 / 24 / 126 on SQuAD. After every verify the frozen policy just stops. That unused state feature is a Milestone-3 win condition for RL, not a reason to drop verify.
6. **Next:** Milestone 3 is cost-aware RL. Extra tools can change a few answers; a learned controller must **select** when that is worth the cost, including when verify says the current evidence is contradictory or only neutral. Preferred single-hop source remains Tevatron/wikipedia-nq if the download is available; this table is the honest SQuAD fallback. Re-run the reward ablation on this slice before putting it next to the ranking table.

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
- `data/processed/slice_meta.json` (`nq_corpus`, `n_nq_anchor`, `retrieval_diag`)

Figures land in `results/figs/` (overall bars plus `policy_by_dataset.png`).
