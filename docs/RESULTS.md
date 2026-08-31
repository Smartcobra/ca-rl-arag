# Results Guide — What Was Produced and How to Read It

**Run this doc describes:** 80k-passage Qwen ranking pilot (300 eval: 150 Hotpot + 150 **Natural Questions** on DPR Wikipedia). Metrics committed in `d456d26` (2026-08-27). Headline artifact: `results/metrics/pilot_summary_default.json`. `slice_meta.json`: `nq_corpus: dpr_wikipedia_w100`, `nq_hf_dataset: Tevatron/wikipedia-nq`, `n_nq_anchor: 0`, **1,450** NQ wiki golds / **847** distinct eval gold articles (not a 7- or 16-article prefix). Section 5 (reward ablation) is the **same corpus**, stratified 100, also from `d456d26`. Section 6 (synthetic) is extractive, 2026-08-07. Section 8 (verifier labels) uses `rule_based_default.jsonl` from this NQ run. The SQuAD fallback table (`e8a4423`) and leaked-NQ table (`2417c43`) are historical.

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
| `reward_ablation_table.json` | Same policy, different reward presets | Justify α/β/γ/λ choices — **same NQ slice**, stratified 100 |

---

## 2. Setup used for the main HuggingFace pilot

| Item | Value |
|---|---|
| Data | HotpotQA distractor + **Tevatron/wikipedia-nq** (`prepare_data.py --hf`) |
| Slice | 100 train / **300 eval** locked (150 Hotpot + 150 NQ) |
| Corpus | **80,000** passages (2,086 Hotpot slice + **1,450** DPR Wikipedia golds + 1,509 DPR negatives + 74,955 unused-Hotpot distractors) |
| BM25 gold recall | Hotpot R@1 0.76 / R@5 **0.927** (11 miss@5); NQ R@1 0.28 / R@5 **0.587** (62 miss@5) |
| Eval limit in pilot | Full eval (`limit: null`) |
| Reward preset | `default` (see `configs/reward_weights.yaml`) |
| Retriever | BM25 |
| Generator | **Qwen2.5-3B-Instruct** |
| Verifier | Lexical NLI |
| Policies | `naive_rag`, `rule_based`, `max_tools` |

These runs validate the **pipeline, costs, and frozen-policy reward ranking**, not SOTA Hotpot/NQ accuracy. NQ golds are real DPR Wikipedia 100-word passages (not `{question} The answer is {gold}`). 847 distinct eval gold articles sit in an 80k Hotpot-heavy index, so first-shot BM25 can fail. Hotpot retrieval is also no longer near-perfect.

Counts and recall: `data/processed/slice_meta.json`.

---

## 3. Main policy comparison (Qwen, 300 eval examples, 80k corpus)

**Comparison in one line:** Hotpot is 59 / 56 / 61. NQ is 41 / 41 / 44. Extra tools move a few answers on both splits, but spend is 1× / 2.9× / 4.3×, so reward ranks **naive > rule > max_tools**. Single-hop is a hard ranking split (~27–29% EM), not a saturated ceiling.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`, `force_yes_no: true`, 80k-passage index, Tevatron NQ).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.654** |
| rule_based | 0.373 | 0.438 | 56/150 | 25 | 0.167 | 4.81e-4 | 0.590 |
| max_tools | **0.407** | **0.485** | **61/150** | 23 | 0.153 | 7.38e-4 | 0.587 |

### Natural Questions (n=150; ranking split — not saturated)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | **0.542** |
| rule_based | 0.273 | 0.352 | 41/150 | 15 | 0.100 | 5.27e-4 | 0.504 |
| max_tools | **0.293** | **0.358** | **44/150** | 18 | 0.120 | 7.55e-4 | 0.464 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | **0.598** |
| **rule_based** | 0.323 | 0.395 | 97/300 | 40 | 5.04e-4 | 4.0 | 0.547 |
| **max_tools** | **0.350** | **0.422** | **105/300** | 41 | 7.47e-4 | 7.0 | 0.526 |

### How to read this table

**Retrieval**
- The 80k unused-Hotpot pool is why first retrieve can fail.
- Hotpot gold is missing from BM25 top-5 on **11/150** items (R@5 0.927).
- NQ gold is missing from BM25 top-5 on **62/150** items (R@5 0.587). That is the distractor pool plus real Wikipedia golds — not answer-anchor leakage. Recall is **below 1.0** on the 80k index, which is the check that the slice is honest.

**Quality**
- **Read Hotpot and NQ, not Overall.** Hotpot sits near 39–41% EM. NQ sits near 27–29% EM. The old leaked-NQ table (146/150 for every policy) is gone.
- Hotpot is still a few-hit race: 59 / 56 / 61. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). `rule_based` vs naive is **2 / 5** (net −3). Too small to call a quality winner.
- NQ: `rule_based` is tied with naive at 41/150 (**1 recovery / 1 regression**). `max_tools` is 44/150 vs 41/150 naive: **5 recoveries / 2 regressions** (net +3). Blind extra retrieves buy three NQ hits and still lose on reward.
- Versus the leaked-NQ 80k run (`2417c43`): overall EM 0.68 → **0.33** because copy-the-anchor is gone. Versus the SQuAD fallback (`e8a4423`): overall EM 0.40 → **0.33** because NQ on DPR Wikipedia is harder than 16 shared SQuAD articles. Hotpot stays in the same 56–61 band.

**Cost / behavior**
- **naive_rag:** 1 retrieve → answer; 1089 ms; 795 tokens.
- **rule_based:** rerank + verify (4 steps; 2.9× $; 1724 ms; 1599 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.3× $; 3387 ms; 2055 tokens). High-cost reference. Latency is up vs the 2k index because each BM25 call scores 80k passages.

**Reward**
- Overall: naive 0.598 > rule 0.547 > max_tools 0.526. Hotpot: 0.654 > 0.590 > 0.587. NQ: 0.542 > 0.504 > 0.464. Extra tools can move a couple of answers; λ/$ still decides the ranking.

**$/correct**
- Overall $/correct is 5.15e-4 / 1.56e-3 / 2.13e-3.

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.365 | 0.647 | −0.017 | 0.035 |
| rule_based | 0.359 | 0.651 | −0.040 | 0.037 |
| max_tools | 0.386 | 0.665 | −0.018 | 0.035 |

- **Q_ans / Q_ground are no longer NQ-inflated.** On Hotpot, Q_ans is 0.419 / 0.406 / 0.446; Q_ground is 0.668 / 0.673 / 0.702. On NQ, Q_ans is 0.311 / 0.313 / 0.326; Q_ground is 0.626 / 0.630 / 0.629 (leaked-anchor NQ was 1.000).
- **Q_cal is negative overall** because both splits are hard. NQ Q_cal is −0.102 / −0.109 / −0.075: abstains and confident wrongs on a real Wikipedia index.

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

**NQ span mismatch** — `natural_questions_eval_13`:

| Field | Value |
|---|---|
| Question | where is human sperm stored in the body |
| Gold | `in the epididymis` |
| Prediction | `epididymis` |
| EM / F1 | 0 / 0.67 |

The span is right. Strict EM fails because gold wants the preposition `in …`. Same pattern as `natural_questions_eval_8` (`close to the poles` vs `near the poles`, F1 0.40). Read F1 and trajectories before calling this a retrieval miss.

When you see `env_rollouts.jsonl` rows like:

```json
{"id": "...", "reward": 0.13, "em": 0.0, "f1": 0.0}
```

it means: that Gym episode ended with a wrong answer and a low reward. Rows with `em: 1.0` / high reward are successful episodes.

---

## 5. Reward-weight ablation results

**Run this section describes:** same Tevatron-NQ 80k Qwen corpus as §3, commit `d456d26` (2026-08-27). Source: `results/metrics/reward_ablation_table.json`.

Fixed policy: **rule_based**, **stratified 100** from the 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.34; Hotpot 17/50, NQ 17/50); only the **scalar reward** changes. This is a subset, not the ranking table.

| Preset | overall reward | Hotpot reward | NQ reward | What it tests |
|---|---:|---:|---:|---|
| correctness_only | 0.362 | 0.373 | 0.351 | Search-R1-like outcome only |
| correctness_grounding | 0.547 | 0.495 | 0.598 | Add grounding (β) |
| correctness_faithfulness_cost | 0.519 | 0.467 | 0.570 | Add cost + hall + act penalties |
| **default** | 0.518 | 0.476 | 0.561 | Full objective (α,β,γ,λ,μ,…) |
| lambda_zero | 0.520 | 0.478 | 0.563 | Remove $ / latency terms |
| high_cost_pressure | 0.456 | 0.417 | 0.495 | Larger λ/μ (cheaper operating point) |

### Interpretation

- EM/F1/$ stay flat across presets because the **policy is frozen**; ablations only change how we **score** trajectories.
- Grounding terms still raise reward (`correctness_only` 0.362 → `correctness_grounding` 0.547), but the jump is smaller than on leaked NQ (0.69 → 0.95) because Q_ground is no longer 1.0 on planted gold.
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
| `dataset` | `hotpot_qa` or `natural_questions` on the current slice (`trivia_qa` / `squad` if those sources load) — required for per-dataset aggregation |
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

**What the current run actually returned** (`rule_based_default.jsonl`, 150 + 150, commit `d456d26`):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA (hard ranking split) | **14** | **31** | **105** |
| Natural Questions (DPR Wikipedia) | **0** | **18** | **132** |

Read this the junior way: the verifier is **not** a yes-man on either split now. On leaked NQ it was support 150/150 because the gold string was planted. On DPR NQ, 18 items are only `neutral` (15/18 abstain, 1/18 EM-correct). On Hotpot it still mixes — about one in ten answers is flagged as a contradiction (8 of those 14 are nonetheless EM-correct), and another ~one in five is only `neutral` (25/31 abstain). That mix is exactly where a verify signal is useful.

**What `rule_based` does with that signal: nothing that changes the search.** On this run, the next action after `verify` was `stop` on **300 / 300** items — including all 14 Hotpot contradictions and all 18 NQ neutrals. It never re-retrieved. It never rewrote the query. (There is a small “if support is very low, retrieve once more” branch in the frozen policy, but it did not fire here, and it does not look at the `contradiction` label anyway.)

So we have a gap:

1. The **signal exists** and is informative on both splits (Hotpot contradictions; NQ neutrals).
2. The **frozen policy does not use it** to recover (no extra retrieve / rewrite after contradiction or neutral).

That gap is precisely where a learned policy should win: see `contradiction` / low support / `neutral` in the state, then spend another retrieve or rewrite only when that looks worth the cost. The frozen baselines cannot show that behavior, so they are not a fair ceiling on what verify is worth.

---

## 9. Takeaways for the paper / next milestone

1. **Pipeline OK:** data → 80k BM25 index → agent actions → NLI verify → cost → multi-component reward → logs on the locked 300-example eval (150 Hotpot + 150 NQ). Preflight requires eval=300 and corpus ≥ 50k before Qwen loads, and rejects leftover NQ answer-anchors.
2. **Reporting contract:** every table is overall + Hotpot + NQ. Overall is mix-weighted. Single-hop is now a ranking split (~27–29% EM), not a saturated ceiling.
3. **The distractor pool did what it was for.** Hotpot R@5 is 0.927 (11 misses). NQ R@5 is 0.587 (62 misses). Overall EM dropped 0.68 → 0.33 vs leaked NQ because copy-the-anchor is gone.
4. **Quality vs cost is now the right story on both splits.** Hotpot 59 / 56 / 61 (max 3 recoveries / 1 regression). NQ 41 / 41 / 44 (max net +3; rule tied). Reward still ranks naive because spend is 1× / 2.9× / 4.3×.
5. **Verify is informative and unused by `rule_based`.** Lexical NLI returns 14 contradiction / 31 neutral / 105 support on Hotpot, and 0 / 18 / 132 on NQ. After every verify the frozen policy just stops. That unused state feature is a Milestone-3 win condition for RL, not a reason to drop verify.
6. **Next:** Milestone 3 is cost-aware RL. Extra tools can change a few answers; a learned controller must **select** when that is worth the cost, including when verify says the current evidence is contradictory or only neutral. This table is the intended Tevatron NQ ranking snapshot. Ablation JSON is already from the same slice.

---

## 10. How to regenerate and refresh this doc

```bash
python scripts/prepare_data.py --hf
# Open data/processed/slice_meta.json before any GPU job.
# Required: nq_corpus=dpr_wikipedia_w100, Tevatron/wikipedia-nq,
# n_nq_anchor=0, many distinct single-hop golds (not 7), NQ recall@5 < 1.0 on 80k.
python scripts/run_pilot.py --run-env-check
python scripts/run_reward_ablation.py
python scripts/plot_results.py
```

Then update numbers in this file and in `EXPERIMENT_LOG.md` from:

- `results/metrics/pilot_summary_default.json` (overall + `by_dataset`)
- `results/metrics/reward_ablation_table.json` and `reward_ablation_by_dataset.json`
- `data/processed/slice_meta.json` (`nq_corpus`, `n_nq_anchor`, `retrieval_diag`)

Figures land in `results/figs/` (overall bars plus `policy_by_dataset.png`).
