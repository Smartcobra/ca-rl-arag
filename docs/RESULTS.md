# Results Guide — What Was Produced and How to Read It

**Run this doc describes:** 80k-passage Qwen ranking pilot (300 eval: 150 Hotpot + 150 **Natural Questions** on DPR Wikipedia). Quality/cost numbers for naive / rule / max_tools are the same slice as `d456d26` (2026-08-27), **rescored 2026-09-04** after closing the `calibration_score` lazy-abstain tautology ([`REWARD_DESIGN.md`](REWARD_DESIGN.md)). Section 5 (reward ablation) is the **same corpus**, stratified 100. Section 6 is REINFORCE: train curve (`train_policy_curve.json`, 2026-09-09) plus the **300-eval** (`learned_default.json`, 2026-09-10). Frozen argmax **collapsed to naive RAG** (retrieve→stop 300/300; predictions identical). `slice_meta.json`: `nq_corpus: dpr_wikipedia_w100`, `nq_hf_dataset: Tevatron/wikipedia-nq`, `n_nq_anchor: 0`, **1,450** NQ wiki golds / **847** distinct eval gold articles. Section 7 (synthetic) is extractive, 2026-08-07. Section 9 (verifier labels) uses `rule_based_default.jsonl` from this NQ run. The SQuAD fallback table (`e8a4423`) and leaked-NQ table (`2417c43`) are historical.

This document describes the **Milestone 2 ranking results** plus Milestone 3 train + 300-eval: where files live, what each metric means, how to interpret the current numbers, and known limitations.

For how to regenerate results, see [`HOW_TO_RUN.md`](HOW_TO_RUN.md).  
For append-only run notes, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).  
Historical leaked-NQ max-tools mechanism (tiny-corpus 2,276-passage run `34e6585`, NQ 148/150): [`NQ_MAX_TOOLS_ANALYSIS.md`](NQ_MAX_TOOLS_ANALYSIS.md). That write-up is **not** this ranking snapshot.

---

## 1. Where results live

```text
results/
├── metrics/                          # Aggregated summaries (committed)
│   ├── pilot_summary_default.json    # Currently naive + learned (rewritten by --policies learned)
│   ├── baseline_default.json         # Naive RAG only
│   ├── rule_based_default.json       # Frozen ranking (not in current pilot_summary)
│   ├── max_tools_default.json
│   ├── learned_default.json          # Frozen REINFORCE on the 300 eval (2026-09-10)
│   ├── train_policy_curve.json       # REINFORCE train-only, 5 epochs, n=100
│   ├── reward_ablation_table.json    # Compact reward-weight sweep (nests by_dataset)
│   ├── reward_ablation_by_dataset.json
│   └── ablation_rule_based_*.json    # Per-preset full summaries
├── checkpoints/                      # Training write path; eval used the .pt then
│   ├── learned_policy.pt             # last-epoch weights (eval loads this)
│   └── learned_policy_best.pt        # best train mean reward (epoch 1)
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
    ├── learned_default.jsonl         # 300 eval rows; all retrieve→stop
    └── env_rollouts.jsonl            # Short Gym env check dumps
```

| File type | Granularity | Typical use |
|---|---|---|
| `pilot_summary_*.json` | Overall + `by_dataset` (Hotpot / NQ) per policy | Comparison table; never cite overall alone. Current file has naive + learned only; rule/max are in their own JSON |
| `learned_default.json` | Frozen REINFORCE on the **300 eval** | Ranking row for the learned policy |
| `train_policy_curve.json` | One object per **train** epoch | Learning signal on the 100. **Not** a ranking number |
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
| Policies (ranking table) | `naive_rag`, `rule_based`, `max_tools`, `learned` |
| Learned policy | 300-eval on disk (`learned_default.json`). Frozen argmax = naive (retrieve→stop 300/300). Train curve is separate. |

These runs validate the **pipeline, costs, and frozen-policy reward ranking**, not SOTA Hotpot/NQ accuracy. NQ golds are real DPR Wikipedia 100-word passages (not `{question} The answer is {gold}`). 847 distinct eval gold articles sit in an 80k Hotpot-heavy index, so first-shot BM25 can fail. Hotpot retrieval is also no longer near-perfect.

Counts and recall: `data/processed/slice_meta.json`.

---

## 3. Main policy comparison (Qwen, 300 eval examples, 80k corpus)

**Comparison in one line:** Hotpot is 59 / 56 / 61 / **59**. NQ is 41 / 41 / 44 / **41**. `learned` (frozen argmax) **is naive RAG** — same answers, retrieve→stop, reward 0.580. Extra tools still move a few answers on rule/max, but spend is 1× / 2.9× / 4.3× / **1×**, so reward ranks **naive = learned > rule > max_tools**.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `limit: null`, `force_yes_no: true`, 80k-passage index, Tevatron NQ; reward/\(Q_{\mathrm{cal}}\) as of 2026-09-04).

### HotpotQA (n=150; ranking split)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.631** |
| rule_based | 0.373 | 0.438 | 56/150 | 25 | 0.167 | 4.81e-4 | 0.570 |
| max_tools | **0.407** | **0.485** | **61/150** | 23 | 0.153 | 7.38e-4 | 0.569 |
| learned | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | 0.631 |

### Natural Questions (n=150; ranking split — not saturated)

| Policy | mean EM | mean F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | **0.529** |
| rule_based | 0.273 | 0.352 | 41/150 | 15 | 0.100 | 5.27e-4 | 0.492 |
| max_tools | **0.293** | **0.358** | **44/150** | 18 | 0.120 | 7.55e-4 | 0.450 |
| learned | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | 0.529 |

### Overall (mix-weighted; do not rank from this)

| Policy | mean EM | mean F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive_rag** | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | **0.580** |
| **rule_based** | 0.323 | 0.395 | 97/300 | 40 | 5.04e-4 | 4.0 | 0.531 |
| **max_tools** | **0.350** | **0.422** | **105/300** | 41 | 7.47e-4 | 7.0 | 0.509 |
| **learned** | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | 0.580 |

### How to read this table

**Retrieval**
- The 80k unused-Hotpot pool is why first retrieve can fail.
- Hotpot gold is missing from BM25 top-5 on **11/150** items (R@5 0.927).
- NQ gold is missing from BM25 top-5 on **62/150** items (R@5 0.587). That is the distractor pool plus real Wikipedia golds — not answer-anchor leakage. Recall is **below 1.0** on the 80k index, which is the check that the slice is honest.

**Quality**
- **Read Hotpot and NQ, not Overall.** Hotpot sits near 39–41% EM. NQ sits near 27–29% EM. The old leaked-NQ table (146/150 for every policy) is gone.
- Hotpot is still a few-hit race: 59 / 56 / 61 / **59**. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). `learned` vs naive is **0 / 0** (same 59 answers).
- NQ: `rule_based` is tied with naive at 41/150. `max_tools` is 44/150. `learned` is **41/150**, same predictions as naive (300/300 identical strings overall).
- Versus the leaked-NQ 80k run (`2417c43`): overall EM 0.68 → **0.33** because copy-the-anchor is gone. Versus the SQuAD fallback (`e8a4423`): overall EM 0.40 → **0.33** because NQ on DPR Wikipedia is harder than 16 shared SQuAD articles. Hotpot stays in the same 56–61 band.

**Cost / behavior**
- **naive_rag:** 1 retrieve → answer; 1046 ms; 795 tokens.
- **rule_based:** rerank + verify (4 steps; 2.9× $; 1765 ms; 1599 tokens).
- **max_tools:** retrieve×3 + rewrite + rerank + verify (7 steps; 4.3× $; 3420 ms; 2055 tokens). High-cost reference.
- **learned:** **same mix as naive** (1 retrieve → stop; 0 verify; 1169 ms; 795 tokens). Frozen argmax never left the naive path.

**Reward**
- Overall: naive **= learned** 0.580 > rule 0.531 > max_tools 0.509. Extra tools can move a couple of answers; λ/$ still decides the ranking. Learned did not spend extra $ and did not recover extra answers.

**$/correct**
- Overall $/correct is 5.15e-4 / 1.56e-3 / 2.13e-3 / **5.15e-4** (learned = naive).

### Reward components (same run, overall)

| Policy | mean Q_ans | mean Q_ground | mean Q_cal | mean P_hall |
|---|---:|---:|---:|---:|
| naive_rag | 0.365 | 0.647 | **−0.137** | 0.035 |
| rule_based | 0.359 | 0.651 | **−0.147** | 0.037 |
| max_tools | 0.386 | 0.665 | **−0.128** | 0.035 |
| learned | 0.365 | 0.647 | **−0.137** | 0.035 |

- **Q_ans / Q_ground are no longer NQ-inflated.** On Hotpot, Q_ans is 0.419 / 0.406 / 0.446; Q_ground is 0.668 / 0.673 / 0.702. On NQ, Q_ans is 0.311 / 0.313 / 0.326; Q_ground is 0.626 / 0.630 / 0.629 (leaked-anchor NQ was 1.000).
- **Q_cal is more negative after the 2026-09-04 fix.** Unjustified abstain is now −0.2 (usable evidence) instead of the old tautology +0.6. Overall Q_cal is −0.137 / −0.147 / −0.128. Hotpot: −0.086 / −0.105 / −0.085. NQ: −0.187 / −0.189 / −0.171. Scoring table: [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

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

**Run this section describes:** same Tevatron-NQ 80k Qwen corpus as §3, rescored 2026-09-04 after the `calibration_score` fix. Source: `results/metrics/reward_ablation_table.json`.

Fixed policy: **rule_based**, **stratified 100** from the 300-file (50 Hotpot + 50 NQ). Same behavior → same EM/F1/$ (EM 0.34; Hotpot 17/50, NQ 17/50); only the **scalar reward** changes. This is a subset, not the ranking table. Presets that include γ \(Q_{\mathrm{cal}}\) (`default`, `lambda_zero`, `high_cost_pressure`) dropped vs the pre-fix table; `correctness_only` / `correctness_grounding` / `correctness_faithfulness_cost` did not (no calibration term).

| Preset | overall reward | Hotpot reward | NQ reward | What it tests |
|---|---:|---:|---:|---|
| correctness_only | 0.362 | 0.373 | 0.351 | Search-R1-like outcome only |
| correctness_grounding | 0.547 | 0.495 | 0.598 | Add grounding (β) |
| correctness_faithfulness_cost | 0.519 | 0.467 | 0.570 | Add cost + hall + act penalties |
| **default** | 0.499 | 0.449 | 0.549 | Full objective (α,β,γ,λ,μ,…) |
| lambda_zero | 0.501 | 0.451 | 0.551 | Remove $ / latency terms |
| high_cost_pressure | 0.431 | 0.382 | 0.479 | Larger λ/μ (cheaper operating point) |

### Interpretation

- EM/F1/$ stay flat across presets because the **policy is frozen**; ablations only change how we **score** trajectories.
- Grounding terms still raise reward (`correctness_only` 0.362 → `correctness_grounding` 0.547), but the jump is smaller than on leaked NQ (0.69 → 0.95) because Q_ground is no longer 1.0 on planted gold.
- Adding calibration (`correctness_faithfulness_cost` 0.519 → `default` 0.499) now *lowers* reward: mean \(Q_{\mathrm{cal}}\) on this subset is negative (−0.13), and lazy abstains are no longer +0.6.
- Cost pressure (`high_cost_pressure`) lowers reward for the same spend — useful later when a learned policy can choose fewer tools.
- On this price card, absolute $ is tiny, so λ effects are modest until you scale prices or tool counts; the relative ordering still moves as designed.

Full weight definitions: [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

---

## 6. REINFORCE: train curve and 300-eval

**Train (2026-09-09).** Source: `results/metrics/train_policy_curve.json`. Same Tevatron-NQ 80k corpus and **fixed** 2026-09-04 calibration rule. Setup: `hidden=16` (261 params), `lr=0.003`, `entropy_coef=0.01`, `reward_preset=default`, `"split": "train"`, `n=100` (60 Hotpot + 40 NQ). Five epochs. The trainer never opened `eval_slice.jsonl`.

Do not put these rewards next to the 300-eval table. Different questions, and training **samples** actions; eval uses **argmax**.

| Epoch | mean reward | mean EM | mean steps | mean retrieve | mean verify |
|---|---:|---:|---:|---:|---:|
| 1 | **0.649** | **0.42** | 4.75 | 1.62 | 0.46 |
| 2 | 0.620 | 0.39 | 4.20 | 1.39 | 0.46 |
| 3 | 0.577 | 0.36 | 4.45 | 1.41 | 0.55 |
| 4 | 0.564 | 0.36 | 4.30 | 1.44 | 0.50 |
| 5 | 0.643 | 0.41 | 4.01 | 1.30 | 0.39 |

Best **train** mean reward is epoch 1 (0.649). Last-epoch reward is 0.643. Train did **not** look like naive: verify stayed 0.39–0.55, steps ~4.

**Eval (2026-09-10).** Source: `results/metrics/learned_default.json`, `results/trajectories/learned_default.jsonl`. Frozen checkpoint, deterministic argmax, `--split eval`, n=300.

| Split | EM | n_correct | steps | retrieve | verify | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.393 | 59/150 | 2.0 | 1.0 | **0.0** | 1.59e-4 | 0.631 |
| Natural Questions | 0.273 | 41/150 | 2.0 | 1.0 | **0.0** | 1.84e-4 | 0.529 |
| Overall | 0.333 | 100/300 | 2.0 | 1.0 | **0.0** | 1.72e-4 | 0.580 |

- **300/300 trajectories are retrieve → stop.** rewrite=0, rerank=0, verify=0, `verify_out=null`.
- Predictions match naive **300/300**. EM/F1/$ match. Reward differs only by wall-clock latency (1169 vs 1145 ms).
- Train entropy hid a naive **mode**. Sampling during training used verify; argmax at test did not. That is why the homework curve is not the exam score.
- The Milestone-3 win condition (re-retrieve / rewrite after `contradiction` or `neutral`) **did not fire** — verify never ran.

`pilot_summary_default.json` was rewritten by `--policies learned` to naive + learned only. Rule / max_tools remain in `rule_based_default.json` / `max_tools_default.json`. Rebuild a four-policy summary with `python scripts/run_pilot.py --run-env-check` if you need one JSON blob.

---

## 7. Synthetic pilot (sanity / offline)

**Run this section describes:** extractive generator, closed synthetic corpus, 16 eval examples, 2026-08-07 (`docs/EXPERIMENT_LOG.md`). Not the 80k Qwen ranking run.

When quality is saturated (extractive EM ≈ 1.0 on a closed fact corpus), reward ranks by cost:

| Policy | mean EM | mean $ | mean reward |
|---|---:|---:|---:|
| naive_rag | 1.00 | 8.3e-5 | 1.525 |
| rule_based | 1.00 | 3.5e-4 | 1.470 |
| max_tools | 1.00 | 4.3e-4 | 1.423 |

This is the cleanest proof that the **cost-aware reward is wired correctly** before you introduce LLM noise.

---

## 8. What each trajectory JSONL field means

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

## 9. Verifier signal vs frozen policy (RL implication)

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

## 10. Takeaways for the paper / next milestone

1. **Pipeline OK:** data → 80k BM25 index → agent actions → NLI verify → cost → multi-component reward → logs on the locked 300-example eval (150 Hotpot + 150 NQ). Preflight requires eval=300 and corpus ≥ 50k before Qwen loads, and rejects leftover NQ answer-anchors.
2. **Reporting contract:** every table is overall + Hotpot + NQ. Overall is mix-weighted. Single-hop is now a ranking split (~27–29% EM), not a saturated ceiling.
3. **The distractor pool did what it was for.** Hotpot R@5 is 0.927 (11 misses). NQ R@5 is 0.587 (62 misses). Overall EM dropped 0.68 → 0.33 vs leaked NQ because copy-the-anchor is gone.
4. **Quality vs cost is now the right story on both splits.** Hotpot 59 / 56 / 61 / 59. NQ 41 / 41 / 44 / 41. Reward ranks naive = learned (0.580) > rule 0.531 > max_tools 0.509 because spend is 1× / 2.9× / 4.3× / 1×.
5. **Lazy abstain is no longer easy reward.** After the 2026-09-04 `calibration_score` fix, overall \(Q_{\mathrm{cal}}\) is −0.137 / −0.147 / −0.128. A learned policy cannot farm +0.6 by refusing whenever gold would have been wrong.
6. **Verify is informative and unused by `rule_based`.** Lexical NLI returns 14 contradiction / 31 neutral / 105 support on Hotpot, and 0 / 18 / 132 on NQ. After every verify the frozen policy just stops. That unused state feature is a Milestone-3 win condition for RL, not a reason to drop verify.
7. **REINFORCE train sampled tools; eval argmax is naive.** Five train epochs: reward 0.649 → 0.564 → 0.643, verify 0.39–0.55, steps ~4. Frozen 300-eval: retrieve→stop **300/300**, predictions identical to naive, reward 0.580. Entropy hid the mode. The verify-on-contradiction win condition did not fire (`verify_out` is null).
8. **Next:** the ranking snapshot now includes a honest `learned` row (tied with naive, cheaper than rule/max). A later controller must actually **select** extra tools on eval, not only while sampling in train. Ablation JSON is from the same slice. Do not cite the train curve as a policy win.

---

## 11. How to regenerate and refresh this doc

```bash
python scripts/prepare_data.py --hf
# Open data/processed/slice_meta.json before any GPU job.
# Required: nq_corpus=dpr_wikipedia_w100, Tevatron/wikipedia-nq,
# n_nq_anchor=0, many distinct single-hop golds (not 7), NQ recall@5 < 1.0 on 80k.
python scripts/run_pilot.py --run-env-check
python scripts/run_reward_ablation.py
python scripts/train_policy.py
python scripts/run_pilot.py --policies learned   # only after learned_policy.pt exists
python scripts/plot_results.py
```

Then update numbers in this file and in `EXPERIMENT_LOG.md` from:

- `results/metrics/pilot_summary_default.json` (currently naive + learned; rule/max in their own JSON)
- `results/metrics/learned_default.json` and `results/trajectories/learned_default.jsonl`
- `results/metrics/reward_ablation_table.json` and `reward_ablation_by_dataset.json`
- `results/metrics/train_policy_curve.json` (`"split": "train"`, n=100)
- `data/processed/slice_meta.json` (`nq_corpus`, `n_nq_anchor`, `retrieval_diag`)

If only the reward formula changed (as on 2026-09-04), re-score the existing trajectories — EM/F1/$ stay; rewrite reward / \(Q_{\mathrm{cal}}\) columns and the ablation table.

Figures land in `results/figs/` (overall bars plus `policy_by_dataset.png`).
