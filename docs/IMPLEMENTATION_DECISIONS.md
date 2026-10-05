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
- **NQ corpus note:** `--hf` loads DPR Wikipedia 100-word passages via `Tevatron/wikipedia-nq` (see 2026-08-24). Answer-anchors are forbidden. Fallback: TriviaQA or SQuAD with real passages.

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
- Eval summaries **must** emit `by_dataset` (`hotpot_qa` plus the loaded single-hop id: `natural_questions` / `trivia_qa` / `squad`) plus overall. Overall is mix-weighted and is not a ranking.

## 2026-08-20 — Per-dataset eval contract

- Trajectory rows already carry `"dataset"`. Aggregation now groups by that field and runs the same means/counts twice.
- Extra counts: `n_examples`, `n_correct`, `n_abstained`, `abstain_rate`.
- Console, `pilot_summary_*.json`, ablation tables, and plots all show Hotpot and NQ separately. Overall stays as a mix-weighted headline after the mix is visible.
- This does **not** make Hotpot gaps significant. NQ is saturated (answer-anchor passages). Ranking waits on the locked 300-example eval (150+150), Hotpot-only.

## 2026-08-21 — Eval grown to 300 (balanced)

- Config lock: `eval_hotpot: 150`, `eval_nq: 150` (train stays 60+40). Balanced mix for the public table; always read Hotpot as the ranking split.
- `python scripts/prepare_data.py --hf` rebuilds `eval_slice.jsonl` / corpus. NQ golds are DPR Wikipedia passages (not answer-anchors). Corpus still grows with unused Hotpot distractors.
- Pilot default is the **full eval file** (no prefix `--limit`). A debug `--limit` is stratified: `round(limit * n_ds / n_total)` per dataset, leftover rounding to hit `limit`. `--limit 40` on 150+150 is ~20+20, never 40 Hotpot + 0 NQ.
- Ablation default is a **stratified 100** from the same 300, labeled as a subset. Not the ranking table.
- Existing Qwen 40-ex metrics stay in `results/` until the 300-example GPU run. Do not rank from them.

## 2026-08-24 (morning) — Lexical NLI is informative; `rule_based` does not act on it

**Run this note describes:** leaked-NQ 80k Qwen ranking pilot, commit `2417c43` (2026-08-23). Superseded later the same day by `e8a4423` (SQuAD), then by `d456d26` (Tevatron NQ). Historical trajectories: `results/trajectories/rule_based_default.jsonl` at that commit.

Trajectory counts (150 Hotpot + 150 NQ, leaked anchors):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA | 14 | 34 | 102 |
| Natural Questions | 0 | 0 | 150 |

Junior reading: `verify` is a “does the evidence agree with this answer?” check. On leaked NQ it always said yes (answer-anchor ceiling). On Hotpot it mixed — 14 contradictions, 34 neutrals — so the signal discriminates on the hard split.

`rule_based` still ignores it for search. After every `verify` the next action was `stop` (300/300). Full write-up of the **current** (Tevatron NQ) counts: `docs/RESULTS.md` §9.

## 2026-08-24 — NQ answer-anchors are label leakage; replaced with DPR Wikipedia (SQuAD fallback)

**Status:** implemented in `scripts/prepare_data.py` / `src/data/wiki_passages.py`. Ranking metrics in `docs/RESULTS.md` (`d456d26`) describe the **Tevatron NQ** run. Do not mix those numbers with leaked-anchor NQ (`2417c43`) or the SQuAD fallback (`e8a4423`).

`--hf` no longer plants `{question} The answer is {gold}`. Primary source is **Tevatron/wikipedia-nq** (DPR 100-word Wikipedia passages, Karpukhin et al. 2020 — the reviewer-expected NQ evidence). The 21M `wiki_dpr` dump is **not** downloaded (Colab). Each NQ item gets its gold Wikipedia passage(s) plus a few DPR negatives; unused Hotpot contexts still grow the shared index to ~80k.

**Fallbacks** (still real passages; never anchors): Tevatron TriviaQA → Tevatron SQuAD → `rajpurkar/squad` article contexts. Preflight rejects any remaining `nq_anchor` rows.

Re-run `python scripts/prepare_data.py --hf` before RL. Ranking scripts will refuse an old anchor corpus.

### What the leaked corpus was (historical)

`nq_open` ships questions and short answers, not Wikipedia passages. Milestone-2 scaffolding planted one synthetic gold per NQ item:

```
{question} The answer is {gold}. According to reference sources, the answer is {gold}.
```

That is **label leakage**. BM25 recall@1 = 1.0, Q_ground = 1.0, P_hall = 0, verify support 150/150, and a policy that learns to stop immediately on NQ for the wrong reason. Acceptable only as M2 pipeline scaffolding; not for GRPO/PPO.

### What `--hf` writes now

| Field | Meaning |
|---|---|
| `nq_corpus` | `dpr_wikipedia_w100` (or `trivia_qa` / `squad` on fallback) |
| `nq_hf_dataset` | `Tevatron/wikipedia-nq` (or fallback id) |
| `n_nq_anchor` | must be 0 |
| `n_nq_wiki` / `n_nq_wiki_neg` | real Wikipedia golds and capped DPR negatives |

After the swap, single-hop recall@k should drop below 1 on the 80k index, and that split can rank stop vs over-retrieve.

## 2026-08-24 (afternoon) — SQuAD fallback ranking pilot (`e8a4423`)

**What ran:** Tevatron/wikipedia-nq was too heavy on the Colab box. `--hf` fell back to `rajpurkar/squad` article contexts. 150 Hotpot + 150 SQuAD, 80k passages, `n_nq_anchor: 0`, only **16** unique SQuAD gold passages in the index.

**Headline numbers** (`pilot_summary_default.json`): Hotpot 59 / 58 / 61. SQuAD 60 / 63 / 60. Overall EM 0.397 / 0.403 / 0.403. Reward naive 0.691 > rule 0.657 > max 0.600. Spend 1× / 3.0× / 4.5×.

**Recall:** Hotpot R@5 0.927 (11 miss). SQuAD R@5 0.633 (55 miss). Overall R@5 0.78.

**Verify** (`rule_based_default.jsonl`): Hotpot 14 contradiction / 34 neutral / 102 support (same mix as leaked NQ, same Hotpot items). SQuAD **0 / 24 / 126** — not a yes-man. After every verify, `stop` 300/300.

**Ablation JSON was not regenerated.** `reward_ablation_table.json` is still the leaked-NQ stratified 100.

Implication: both splits now rank. Milestone 3 can condition on verify. Preferred single-hop source remains Tevatron NQ; that download succeeded on 2026-08-27 (`d456d26`).

## 2026-08-27 (afternoon) — Tevatron NQ ranking pilot (`d456d26`)

**What ran:** `datasets<3.0` + `trust_remote_code` unblocked Tevatron/wikipedia-nq. `--hf` did **not** fall through to SQuAD. 150 Hotpot + 150 NQ, 80k passages, `n_nq_anchor: 0`, **1,450** NQ wiki golds / **847** distinct eval gold articles (not 7). NQ BM25 R@5 **0.587** (below 1.0) on the 80k index.

**Headline numbers** (`pilot_summary_default.json`): Hotpot 59 / 56 / 61. NQ 41 / 41 / 44. Overall EM 0.333 / 0.323 / 0.350. Reward naive 0.598 > rule 0.547 > max 0.526. Spend 1× / 2.9× / 4.3×.

**Recall:** Hotpot R@5 0.927 (11 miss). NQ R@5 0.587 (62 miss). Overall R@5 0.757.

**Verify** (`rule_based_default.jsonl`): Hotpot 14 contradiction / 31 neutral / 105 support. NQ **0 / 18 / 132** — not a yes-man. After every verify, `stop` 300/300.

**Ablation JSON was regenerated** on this slice (stratified 100, EM 0.34). Grounding still lifts reward (0.362 → 0.547) but less than leaked NQ.

Implication: both splits rank on real Wikipedia evidence. Milestone 3 can condition on verify. This is the intended ranking snapshot. Reward/\(Q_{\mathrm{cal}}\) on disk were later rescored (2026-09-04); EM/F1/$ in this block are still current.

## 2026-08-27 — Adaptive-RAG baseline settled; experiments-section sentence

**Pick (locked):** Adaptive-RAG as the required non-RL adaptive baseline. Self-RAG is out of V1. CRAG is an optional later critique-slot stand-in, not the required baseline. Reproduce the *query-complexity classifier idea* on our Qwen + BM25 + Hotpot/single-hop stack; do not port `starsuzi/Adaptive-RAG` wholesale.

**Experiments-section sentence (keep verbatim):**

> Adaptive-RAG routes each query before seeing retrieval quality, which makes it the open-loop contrast our closed-loop controller is supposed to beat on the quality–cost Pareto.

**GRASP public code:** **No.** Gandhi et al. (arXiv:2607.10463) does not link a repo, checkpoint, or Hugging Face collection for GRASP itself (the only HF link in the paper is Search-R1). GitHub / HF papers / Papers with Code have no official artifact. Unrelated repos named GRASP exist (e.g. PKU-ML graph reasoning). A GRASP baseline would be a reimplementation, not a checkpoint load.

## 2026-09-04 — `calibration_score` lazy-abstain tautology closed; metrics rescored

**Bug:** `return 0.6 if (not has_evidence or not correct) else -0.2` always returned +0.6 on abstain, because `not correct` is always true after the refused-solvable check. Combined with \(P_{\mathrm{hall}}=0\) on abstain, a learning policy could farm reward by always refusing.

**Fix** (`src/rewards.py`, `tests/test_rewards.py`): +0.6 only when evidence is empty, mean retrieval `score` `< 3.0`, or `verify_out.label == "contradiction"`. Otherwise abstain is −0.2. Scoring table: `docs/REWARD_DESIGN.md`.

**Rescore:** same Tevatron-NQ 80k trajectories. EM/F1/$ unchanged. Reward naive 0.598 → **0.580**, rule 0.547 → **0.531**, max 0.526 → **0.509**. Overall \(Q_{\mathrm{cal}}\) −0.017 / −0.040 / −0.018 → **−0.137 / −0.147 / −0.128**. Ablation presets with γ dropped (default 0.518 → 0.499); presets without γ did not. Ranking still naive > rule > max.

Do not train GRPO/PPO against the pre-fix \(Q_{\mathrm{cal}}\).

## 2026-09-05 — Tiny REINFORCE trainer (train-only)

**Choice:** Milestone 3 starts as vanilla REINFORCE on a **261-parameter** MLP (`10 → 16 → 5`) in `src/policies/learned.py`, trained by `scripts/train_policy.py`.

**Not chosen (yet):** GRPO/PPO, a learned critic, or query/passage text in the policy.

**Data lock:** train on `train_slice.jsonl` only (60 Hotpot + 40 NQ = 100). The script has no `--split` flag and refuses any path whose filename contains `eval`. The locked 300-example eval is not opened during training. `--limit` is a stratified cap on **train**.

**Why tiny:** 100 trajectories cannot support a wide net. Hidden width is capped at 32 (`ValueError` above that). Observation is the existing env vector (including verify `support` / `contradiction`). Reward is the **fixed** 2026-09-04 calibration rule.

**Update:** per-episode REINFORCE with an EMA reward baseline and entropy 0.01. Sparse terminal reward from `AgenticRAGEnv`. Learning curves: `results/metrics/train_policy_curve.json` (`default`) plus `train_policy_curve_correctness_only.json` / `train_policy_curve_lambda_zero.json` / `train_policy_curve_frontier_act*.json` / `train_policy_curve_frontier_lambda0.json` / `_lambda20.json`. Checkpoints now on disk: `learned_policy.pt` (`default` ranking) plus `learned_policy_correctness_only.pt` / `learned_policy_lambda_zero.pt` / `learned_policy_frontier_act0.pt` / `_act005.pt` / `_act02.pt` / `learned_policy_frontier_lambda0.pt` / `_lambda20.pt` (and `_best.pt` / `_trainer.pt` copies). `--curve` / `--checkpoint` keep those runs from overwriting each other. The retargeted 40-epoch `learned_policy_frontier_act02.pt` is now on disk (2026-09-24); it overwrote the 5-epoch act02 checkpoint.

**Eval (not training):** `run_pilot.py` default `--policies` is `naive_rag,rule_based,max_tools,learned`. The checkpoint is loaded frozen (`deterministic` argmax) and run through the same `evaluate_agent` path, so `pilot_summary_*.json` gets a fourth `by_dataset` block. Missing checkpoint → skip `learned` (or exit if it is the only policy). The trainer still never opens the 300. Train mean reward is not a ranking number.

**First train run (2026-09-09):** `train_policy_curve.json` is on disk. Five epochs on n=100: mean reward 0.649 / 0.620 / 0.577 / 0.564 / 0.643. Mean verify 0.46 / 0.46 / 0.55 / 0.50 / 0.39. Mean steps 4.75 → 4.01.

**First 300-eval (2026-09-10):** `learned_default.json` + `learned_default.jsonl`. Frozen argmax is retrieve→stop **300/300**, verify 0, predictions identical to naive, reward 0.580. Train entropy hid a naive mode. Details: `docs/RESULTS.md` §6.

## 2026-09-12 — Collapse to naive is the reward, not the network

The frozen argmax tying naive RAG is not a failed MLP. Under the default weights the network read the landscape correctly and picked the cheapest corner.

Arithmetic from the committed 300-eval (`RESULTS.md` §3, same Tevatron-NQ 80k slice):

| Term | naive | max_tools | Δ (max − naive) |
|---|---:|---:|---:|
| steps | 2.0 | 7.0 | +5 |
| \(P_{\mathrm{act}}\) (`0.02 * max(0, n-1)`) | 0.02 | 0.12 | **+0.10** |
| mean $ | 1.72e-4 | 7.47e-4 | +5.75e-4 |
| \(\lambda(C_{\mathrm{tok}}+C_{\mathrm{ret}})\), \(\lambda=2\) | 3.4e-4 | 1.5e-3 | **+0.001** |
| \(Q_{\mathrm{ans}}\) | 0.365 | 0.386 | +0.021 |
| \(Q_{\mathrm{ground}}\) | 0.647 | 0.665 | +0.018 |
| \(Q_{\mathrm{cal}}\) | −0.137 | −0.128 | +0.009 |
| quality \(\alpha\Delta Q_a+\beta\Delta Q_g+\gamma\Delta Q_c\) | | | **≈ +0.028** |
| mean reward | 0.580 | 0.509 | **−0.071** |

Using the full tool suite costs an extra **0.10** in the flat action tax plus about **0.001** in real dollars. The best quality those tools buy on this eval is about **0.028** (max_tools recovering a couple of answers — Hotpot net +2, NQ net +3 — plus a little grounding). Spending therefore loses by roughly **0.07** on average. Even perfectly targeted spending barely breaks even: the quality ceiling is smaller than the tax on five extra steps. After 500 train episodes the 261-parameter MLP chose retrieve→stop. It solved the problem we gave it. That problem is not the dollar-aware control problem we meant.

**Cost is currently ~99% the flat action tax and ~1% real dollars.** Of the extra 0.101 paid to run max_tools, \(0.10 / 0.101 \approx 99\%\) is \(P_{\mathrm{act}}\) and \(0.001 / 0.101 \approx 1\%\) is \(\lambda \times \$\). The FinOps price card is almost decorative. Independently: the `lambda_zero` ablation (same rule_based trajectories, \(\lambda=\mu=0\), \(P_{\mathrm{act}}\) kept) moves mean reward only 0.499 → 0.501.

`act_penalty: 0.02` was designed as an anti-loop guard. `reward_weights.yaml` says “anti-loop, not anti-retrieve”; `REWARD_DESIGN.md` says it is “deliberately ≪ cost of one useful retrieve.” At 0.02 per action after the first it dominates everything. One extra retrieve’s measured dollar hit is \(\lambda \times \sim 1.7\times 10^{-4} \approx 3\times 10^{-4}\); the action tax on that same step is 0.02 — about **60×** larger. For a paper whose claim is dollar-aware control, that balance is backwards.

This is a reward-design finding, not a trainer or architecture finding. Do not retune the MLP to “use more tools” while this scalar still makes tools −EV. A later weight change (shrink \(P_{\mathrm{act}}\), or make it a true loop tax rather than a per-step tax) is a new experiment; it is not a code bug in `src/rewards.py`.

## 2026-09-13 — Free-cost sanity: trainer works; \(P_{\mathrm{act}}\) is the pin

Trained and scored two new checkpoints on the same 100 / 300 split. Did not overwrite `learned_policy.pt`.

**`correctness_only`** (cost free): train steps 4.66 → 5.84, verify 0.50 → 1.20. Frozen 300-eval is **8.0 steps / 3.0 retrieve / 2.0 rewrite / 2.0 verify on 300/300** (step cap). EM 0.340 (102/300) vs naive 100/300 (Hotpot 60 vs 59, NQ 42 vs 41; 9 recoveries / 7 regressions). Reward 0.378 vs naive 0.365. Sources: `learned_correctness_only.json`, `train_policy_curve_correctness_only.json`, `learned_policy_correctness_only.pt`.

**`lambda_zero`** (λ=μ=0, \(P_{\mathrm{act}}\) kept): train looks like `default` (steps ~4, verify 0.39–0.55). Frozen 300-eval is retrieve→stop **300/300**, predictions identical to naive, reward 0.581. Sources: `learned_lambda_zero.json`, `train_policy_curve_lambda_zero.json`, `learned_policy_lambda_zero.pt`.

The 09-12 arithmetic predicted this split. The network was never stuck at two steps; the default scalar made two steps optimal. Turning off dollars is not the fix — \(P_{\mathrm{act}}\) is.

## 2026-09-13 — λ=80 frontier: three learned policies, one naive point

Trained `frontier_act0` / `frontier_act005` / `frontier_act02` (same quality terms as `default`, \(\lambda=80\)). Frozen 300-eval is retrieve→stop **300/300** on every knob. Predictions identical to naive. EM 0.333 / $ 1.72e-4 / 0 verify.

Train sampling still used tools, then shrank (act0: steps 4.75 → 2.93, verify 0.46 → 0.14; act02 last-epoch on disk: 3.11 / 0.23 / 0.642). Sources: `learned_frontier_act*.json`, `train_policy_curve_frontier_act*.json`, `frontier_sweep_table.json`. Checkpoints synced into this checkout 2026-09-15/16.

Arithmetic: extra max_tools $ × 80 ≈ **0.046** vs quality ≈ **0.028**. Zeroing \(P_{\mathrm{act}}\) is not enough while λ is this large. The missing frontier is a **λ** problem. The only eval that left two steps remains `correctness_only` (λ=0 and \(P_{\mathrm{act}}=0\)).

## 2026-09-15 — Frontier retarget: cross break-even; 40 epochs; greedy eval

**Choice:** replace the three λ=80 `act_penalty` presets with a family that actually straddles the extra-spend break-even line, and stop training 5-epoch curves.

**Presets** (`configs/reward_weights.yaml`): `frontier_lambda0` (λ=0, \(P_{\mathrm{act}}=0\)), `frontier_lambda20` (λ=20, \(P_{\mathrm{act}}=0\)), keep `frontier_act02` (λ=80, \(P_{\mathrm{act}}=0.02\)). Retired: `frontier_act0`, `frontier_act005`. Arithmetic: extra max_tools $ \(\approx 5.8\times 10^{-4}\); 20 × that ≈ **0.012 < 0.028** quality; 80 × that ≈ **0.046 > 0.028**.

**Trainer:** `policy.learned.epochs` is **40** (inside 30–50). After every sampled epoch, `train_policy.py` rolls out argmax on the same 100 train examples and logs `eval_reward` / `eval_n_steps` / `eval_n_verify` (`eval_split: train_greedy`). `_best.pt` is the best greedy reward. The trainer still never opens `eval_slice.jsonl`. Frozen 300-eval remains a separate `run_pilot.py --policies learned` job.

**Why both:** five epochs left every curve in the sampled mode. Eval argmax dropped verify. Without a greedy column on the next sweep we cannot tell whether a collapse is the reward or the training.

Notebook: `notebooks/Frontier_Cost_Pressure_CA_RL_ARAG_v2.ipynb`. 2026-09-15/16 Colab still trained `frontier_act0` / 5 epochs; those `.pt` files stay local as the failed family. **2026-09-23:** `learned_frontier_lambda0.json` / `_lambda20.json` are on disk (see the dated block below).

## 2026-09-23 — λ=0 / 20 family scored

Trained and scored `frontier_lambda0` and `frontier_lambda20` for 40 epochs on the same 100 / 300 split. Did not overwrite `learned_policy.pt` or the 5-epoch λ=80 checkpoints.

**`frontier_lambda0`** (λ=0, \(P_{\mathrm{act}}=0\)): last sampled 4.98 steps / 1.04 verify / reward 0.720; last greedy 6 / 1 retrieve / 2 verify. Frozen 300-eval is **6.0 steps / 1.0 retrieve / 2.0 rewrite / 2.0 verify on 300/300**. EM 0.347 (104/300) vs naive 100/300 (Hotpot 53 vs 59, NQ 51 vs 41). $ 3.41e-4. Sources: `learned_frontier_lambda0.json`, `train_policy_curve_frontier_lambda0.json`, `learned_policy_frontier_lambda0.pt`.

**`frontier_lambda20`** (λ=20, \(P_{\mathrm{act}}=0\)): last sampled 4.01 / 0.52 / 0.711; last greedy 4 / 3 retrieve / 0 verify. Frozen 300-eval is **4.0 steps / 3.0 retrieve / 0 rewrite / 0 verify on 300/300**. EM 0.333 (100/300), same 59+41 split as naive, $ 2.72e-4. Not retrieve→stop. Sources: `learned_frontier_lambda20.json`, `train_policy_curve_frontier_lambda20.json`, `learned_policy_frontier_lambda20.pt`.

`frontier_act02` at 40 epochs is now on disk (2026-09-24): retrieve→stop 300/300, same exam as naive. `frontier_sweep_table.json` and `frontier_em_usd.png` were regenerated the same day.

The EM numbers in this block are the **last** epoch. They are not the best-greedy checkpoints. Read with 2026-09-26.

## 2026-09-24 — λ=80 `act02` 40-epoch exam

Trained and scored `frontier_act02` for 40 epochs on the same 100 / 300 split. Overwrote the 5-epoch act02 `.pt` / JSON / curve (backup curve in `partial_curve_backup/`).

Last sampled 2.09 steps / 0 verify / reward 0.695. Greedy retrieve→stop on all 40 epochs. Frozen 300-eval is **2.0 steps / 1.0 retrieve / 0 verify on 300/300**, EM 0.333 (100/300), $ 1.72e-4, Hotpot 59 / NQ 41. Sources: `learned_frontier_act02.json`, `train_policy_curve_frontier_act02.json`, `learned_policy_frontier_act02.pt`.

Then `plot_results.py --frontier` rewrote `frontier_sweep_table.json` and `frontier_em_usd.png` from the λ=0 / 20 / 80 exams. For λ=80, last epoch and best greedy epoch are the same path. For λ=0 and λ=20 they are not. See 2026-09-26.

## 2026-09-26 — Last checkpoint is a fixed recipe, not a controller

The 40-epoch exams above scored the last `.pt`. `_best.pt` is the best greedy reward on the same 100 training questions. For λ=0 and λ=20 those are different policies. All three frozen 300-evals take one action sequence on every question. They do not use passage count, BM25 score, or the verify label. This run is a flat frontier of fixed recipes. It is not the adaptive-controller figure. (**2026-10-01:** “do not use BM25 score” understates it — that channel was saturated to a constant, so they *could* not. **2026-10-03:** after the fix, λ=20 does branch on it.)

**Checkpoint pick.** `train_policy_curve_frontier_lambda0.json`: best greedy reward is epoch 25, retrieve→stop, `eval_reward` 0.72947, train EM 0.43. The 6-step path (rewrite, rewrite, retrieve, verify, verify, stop) first shows up at epoch 31, reward about 0.712, train EM 0.40. Epochs 34 and 37 flip back to a 4-step path at the same 0.40. The 300-eval used epoch 40: 104/300 versus naive 100/300. On the 100 training questions that recipe is 3 worse (40 vs 43). On the 300 it is 4 better. That is noise.

`train_policy_curve_frontier_lambda20.json`: best greedy reward is epoch 15, retrieve→stop, `eval_reward` 0.72608, train EM 0.43. Epoch 40 is three retrieves then stop, reward 0.72335. Epochs 38–39 had already returned to retrieve→stop. The 300-eval used epoch 40.

**λ=20 spends for the same passages.** `learned_frontier_lambda20.jsonl` is `retrieve, retrieve, retrieve, stop` on 300/300. The three retrieves return the same top passage ids on every question. Predictions match `learned_frontier_act02.jsonl` (naive) on 300/300, so the answers stay 59 Hotpot + 41 NQ. Two wasted retrieves move greedy reward by about 0.0027 (0.72608 → 0.72335). That is smaller than the gap between recipes, so training never treats the repeat as a mistake.

**The observation is ignored.** `learned_frontier_lambda0.jsonl` is the 6-step recipe on 300/300. Final `verify_out` is contradiction 14 / neutral 55 / support 231. After a non-support verify, the next action is never retrieve or rewrite (0 of 138 step events). `learned_frontier_act02.jsonl` is retrieve→stop on 300/300. Greedy keeps flipping because these recipes sit within about 0.02 train reward of each other.

**The per-question gap is the result.** Against naive (`act02`; λ=20 matches it answer for answer), the λ=0 recipe fixes 6 Hotpot answers and breaks 12, and fixes 18 NQ answers and breaks 8. Oracles on the same 300 ids:

| Pick | Correct / 300 |
|---|---|
| Naive on Hotpot, λ=0 recipe on NQ | 110 |
| Better of naive or λ=0, per question | 124 |
| Better of naive, λ=0, and max_tools | 127 |
| Better of naive, λ=0, max_tools, rule_based, and `correctness_only` | 130 |

Single policies only span 100 (naive) to 105 (max_tools). The 110 mix costs $2.60e-4 per question, about a third of max_tools ($7.47e-4, 7 steps). Fixed recipes fight over about 5 points. A controller that actually reads the observation has 24 points of headroom (100 → 124) before any new tool.

**Do not cite** epoch-40 λ=0 EM 0.347 or epoch-40 λ=20 as learned frontier points. The selected exam is `_best.pt`: epoch 25 for λ=0 and epoch 15 for λ=20. On the 300, both are retrieve→stop (100/300, same dollars as naive). The last-epoch rows stay in `frontier_sweep_table.json` beside them. Hollow markers are the ceilings 110, 124, and 127.

**Verify softmax (the 2026-09-10 check).** `learned_policy_frontier_lambda0.pt` (epoch 40), one real post-verify observation, support=1 versus contradiction=1. Raw probabilities: support 0.106 / 0.280 / 0.032 / 0.374 / 0.209 and contradiction 0.099 / 0.274 / 0.036 / 0.379 / 0.212 (retrieve, rewrite, rerank, verify, stop). Max gap 0.0068. They are not identical at two decimals. Argmax is verify either way, so the bit does not change the action. Source: `results/metrics/verify_sensitivity_lambda0.json`.

**Reproducing it (added 2026-10-03).** `scripts/verify_softmax_probe.py` regenerates that file; run with no arguments it recomputes the probe and diffs the committed JSON, exiting non-zero on a mismatch. The probe point is not hand-picked: it is the first decision in `learned_frontier_lambda0.jsonl` whose observation already carries a verify result (`counts.verify >= 1`), which is `hotpot_eval_5a8b57f25542995d1e6f1371` step 4. Two wrinkles are worth knowing. The checkpoint is **10-d**, so `LearnedPolicy.load` refuses it and the script reads the state dict directly after checking the width. And the 10-d vectorizer is **replayed inside the script** (a copy of `src/rag_env.py` at commit `23dc131`) rather than imported: the live one is 16-d with z-scored BM25 channels, so pushing a v2 checkpoint through it would answer a different question, and keeping the live one 10-d-compatible would hold current code hostage to a frozen number. That copy must not be refactored to track `vectorize_structured_obs`. The JSON was also **untracked until now** — both docs cited a file a fresh clone did not have.

Sources: `train_policy_curve_frontier_lambda0.json`, `train_policy_curve_frontier_lambda20.json`, `learned_frontier_lambda0.jsonl`, `learned_frontier_lambda20.jsonl`, `learned_frontier_act02.jsonl`, `learned_frontier_lambda0_best.json`, `learned_frontier_lambda20_best.json`, `max_tools_default.jsonl`, `rule_based_default.jsonl`, `learned_correctness_only.jsonl`.

## 2026-09-26 — v3 train: validation pick, naive anchor, richer observation

This sweep is `notebooks/Frontier_Cost_Pressure_CA_RL_ARAG_v3.ipynb` and `configs/frontier_v3.yaml`. It leaves the v2 checkpoints, curves, and `frontier_em_usd.png` in place. The scored exams are the 2026-09-28 block.

**Validation slice.** Train is 300 Hotpot + 200 NQ. Validation is the next 110 Hotpot + 70 NQ from the same HuggingFace train split (180 questions). `_best.pt` is the highest greedy reward on that slice (`eval_split: valid_greedy`). The 300 eval ids are locked in `tests/fixtures/locked_eval_ids.json`. `scripts/build_train_valid.py` checks them and does not rewrite `eval_slice.jsonl`.

**Advantage.** `advantage = sampled reward − naive reward` on the same question. The naive retrieve-then-stop reward is cached per preset at `results/metrics/naive_reward_cache_<preset>.json`, because λ changes the scalar. The EMA baseline is still written into the trainer file so `--resume` keeps its schema. It is not the advantage. A group of K samples plus this anchor is Milestone 4. This trainer uses one sample.

**Observation.** The vector is 15-d (341 parameters at hidden 16). Appended to the old 10 dims: tanh(top-1 BM25), tanh(top-1 − top-2), and a verify one-hot for support, contradiction, and neutral (all zero before verify). There is no dataset id. Old 10-d `.pt` files do not load into this head.

## 2026-09-28 — v3 exams: one λ=0 recipe, λ=20 and λ=80 stay retrieve→stop

All three presets finished 20 epochs. `_best.pt` is the max greedy reward on the 180 (`eval_split: valid_greedy`). The `v3` JSON is that checkpoint. The `v3last` JSON is epoch 20. The v2 files are unchanged. `frontier_sweep_table_v3.json` and `frontier_em_usd_v3.png` are not in this checkout.

**`frontier_lambda0`.** Best greedy is epoch 11, valid reward 0.61941, valid EM 0.333 (60/180), 6 steps / 3 retrieve / 0 verify. Frozen 300-eval is `retrieve, rewrite, rewrite, retrieve, retrieve, stop` on **300/300**. EM **0.343 (103/300)**, F1 0.421, $ 4.21e-4, eval reward under this preset 0.622. Hotpot 60/150, NQ 43/150. Against this run's retrieve→stop: Hotpot 1 recovery / 1 regression, NQ 8 / 5. The second retrieve changes the top-5 on 106/300 and the top-1 on 52/300. The third retrieve copies the second on 300/300. Epoch 20 is `retrieve, rewrite, rewrite, stop` on 300/300, EM 0.333 (100/300), $ 3.21e-4, and those predictions match retrieve→stop on 300/300. Valid reward at epoch 20 is 0.59292. Epoch 1 is the only greedy epoch with a non-integer step count (2.644 / 1.644 retrieve, reward 0.59356). Sources: `learned_frontier_lambda0_v3.json` / `_v3last.json`, `train_policy_curve_frontier_v3_lambda0.json`.

**`frontier_lambda20`.** Best greedy is epoch 7, valid reward 0.59043, retrieve→stop, valid EM 0.322 (58/180). The best 3-retrieve greedy epoch (epoch 2) is 0.0026 under that. Frozen best and last exams are retrieve→stop on **300/300**, same 300 predictions, EM 0.333 (100/300), Hotpot 60 / NQ 40, $ 1.72e-4, eval reward 0.598. Sources: `learned_frontier_lambda20_v3.json` / `_v3last.json`, `train_policy_curve_frontier_v3_lambda20.json`.

**`frontier_act02`.** Greedy is retrieve→stop on all 20 epochs. Best is epoch 11 (valid reward 0.56024). Best and last exams share the λ=20 answers: EM 0.333 (100/300), $ 1.72e-4, eval reward 0.568. Sources: `learned_frontier_act02_v3.json` / `_v3last.json`, `train_policy_curve_frontier_v3_act02.json`.

Sampled train still calls verify (λ=0 last epoch 0.38, λ=20 0.63, λ=80 0.04). Every frozen exam has verify 0, so the new verify one-hot is zero on the 300. These retrieve→stop predictions match the v2 naive exam (`learned_frontier_act02.jsonl`) on 291/300; 20 questions have different BM25 top ids, and the split is 60/40 against the ranking row 59/41. Overall EM stays 100. The +3 is against this run's retrieve→stop. The union of selected λ=0 and that retrieve→stop is 109/300.

## 2026-10-01 — BM25 score channels were a constant; z-score them

**Bug.** The observation squashed BM25 with `tanh(score / 5)`. On the v3 λ=0 exam (`learned_frontier_lambda0_v3.jsonl`, 300 questions, raw BM25, no rerank) the top-1 score runs from **26.2 to 122.3**, median **49.3**. `tanh(49.3 / 5)` is 1.0000. `tanh(26.2 / 5)` is 0.9999. In float32, 192 of the 300 top-1 features are exactly 1, and the rest are 0.9999. The mean of the top-5 bottoms out at 22.4, so that channel is 0.9997 or 1.0. A network cannot branch on a feature that does not change. The paper sentence “the policy does not branch on the retrieval score” is true, and the honest version is “it could not, the feature was a constant.”

The gap was the only score input that moved. `tanh(gap / 5)` runs from 0 to 1, median 0.57, standard deviation 0.36. In v2 the mean score was the only score input, so after the first retrieve the v2 policy saw almost the same vector on every question.

**Fix.** Replace `tanh(score / 5)` with a z-score from the training slice, then clip to [−3, +3]. The same transform is applied to four numbers: the mean of the top-5, the top-1, the top-1 − top-2 gap, and the minimum of the top-5. The minimum is a new last dimension. The vector is now **16-d** (357 parameters at hidden 16; 709 at hidden 32). Empty retrieval still writes 0 in those channels, so “we have not searched” is not coded as a very negative score. A real gap of 0 (a tie) is standardized; a missing second passage is left at 0.

Constants are frozen in `data/processed/bm25_score_norm.json`. They are the first BM25 top-5 on `train_slice.jsonl` (60 Hotpot + 40 NQ) against the 80k corpus. Population standard deviation. This is the 100-question slice on disk, not the 500-question v3 train prefix (that file is not in this checkout).

| Channel | Train mean | Train std | Train min…max |
|---|---:|---:|---|
| mean of top-5 | 50.95 | 26.85 | 22.7 … 186.7 |
| top-1 | 61.47 | 38.59 | 24.0 … 294.7 |
| gap | 9.54 | 15.13 | 0 … 98.8 |
| top-5 minimum | 46.04 | 23.06 | 20.0 … 142.9 |

Two training questions have a top-1 more than 3 standard deviations up. The largest z is 6.04 (`hotpot_train_5a8a2ebc5542996c9b8d5e33`, passage “Royal Commission into Drug Trafficking”) and the clip pulls it back to 3. On the 300, nothing hits the clip. After the fix the four channels have standard deviation about **0.56 / 0.49 / 0.50 / 0.61** (mean, top-1, gap, top-5 min). `tests/test_score_features.py` vectorizes those 300 trajectories and fails if any of the four is below 0.05.

The other channels (step counters, budget fraction, verify one-hot) are not in that check. This exam is one action sequence, so those channels are constant for a different reason. The budget fraction only moves in the fourth decimal.

**What this does not do.** The 2026-09-28 exams stay the record of the 15-d tanh head. A 10-d or 15-d checkpoint does not load. The next train has to be a new checkpoint. The branching claim is fair only after that train.

### Calibration `+0.6` when the mean score is below 3

`calibration_score` pays +0.6 for an abstain when evidence is empty, **or** the mean retrieval score is `< 3.0`, **or** verify says contradiction. The `3.0` cutoff is the same number `rule_based` uses for “strong enough to rerank and verify.”

On the logged 300 it never sees a mean below 3:

| File | What the score is | Lowest mean with evidence |
|---|---|---:|
| `learned_frontier_lambda0_v3.jsonl` | raw BM25 | 20.42 |
| `learned_frontier_act02.jsonl` | raw BM25, first retrieve | 20.38 |
| `rule_based_default.jsonl` | includes the lexical rerank | 7.27 |
| `max_tools_default.jsonl` | includes the lexical rerank | 7.27 |

So the “mean score < 3” branch is dead on this corpus. It was written for a different scale: a score that lives near 0–3, not Okapi BM25 on 80k passages (top-1 around 25–120) and not the reranker, whose means still bottom out near 7. The unit test that passes `score: 1.0` still describes the rule. A real mean does not take that path: abstaining on the weakest raw mean in the v3 exam (20.42) scores **−0.2**, the lazy-refuse penalty.

The other two ways to earn +0.6 still work. No passages is +0.6. A contradiction label is +0.6, and Hotpot does produce contradictions. Those are not dead.

The threshold is left at 3.0. Moving it would rescore every old trajectory and would also change `rule_based`, which takes the “strong” branch on every question for the same reason (that is why rule-based rewrite count is 0). A later experiment can retarget the cutoff onto the z-score scale. That is a reward change, not part of this observation fix.

## 2026-10-01 — One index per table; frozen baselines must be re-scored on v3

**Bug.** Building the v3 train (500) and validation (180) slices merged their gold passages into `corpus.jsonl`. BM25 scores against the whole corpus, so adding passages changes the ranking for **every** question, including the locked 300. The v3 exams therefore ran on a different index from the frozen rows.

The evidence was already in the 2026-09-28 block: the v3 retrieve→stop predictions match the v2 naive exam on **291/300**, 20 questions have different BM25 top ids, and that run's retrieve→stop splits Hotpot 60 / NQ 40 against the v2 ranking row 59 / 41. Overall EM is 100 either way, which is why this was easy to miss.

Comparing selected λ=0 against *this run's* retrieve→stop was correct and stays correct. What was wrong is everything that put a v3 row next to an old-index row:

- `frontier_sweep_table*.json` / `frontier_em_usd*.png` drew the frozen squares from `baseline_default.json`, `rule_based_default.json`, `max_tools_default.json`. Those are the old index, and `--frontier-tag v3` did not change them.
- The controller ceilings (110 / 124 / 127) were built from `learned_frontier_act02.jsonl`, `learned_frontier_lambda0.jsonl`, and `max_tools_default.jsonl` — all old index — and were then quoted beside v3 points. A per-question oracle across two indexes is comparing answers to two different retrievals.
- The README pilot tables put the v3 learned rows in the same table as old-index naive / rule / max.

**Rule.** Every number in one table or figure comes from one index.

**Enforcement, not convention.**

1. `run_pilot.py --artifact-suffix` now forks **every** policy, not just `learned`. It previously ignored the suffix for naive / rule / max, so a v3 baseline run would have silently overwritten the old-index files that the committed figures were built from. `baseline_default_v3.jsonl`, `rule_based_default_v3.jsonl`, and `max_tools_default_v3.jsonl` now sit beside the originals.
2. `plot_results.py --frontier-tag v3` reads `*_v3.json` frozen anchors and `*_v3.jsonl` ceiling pools. If any is missing it **exits** with the command that produces them instead of falling back to the untagged files. That fallback was the bug.
3. Every pilot artifact records a corpus fingerprint (`n_passages` plus a sha1 over the sorted passage ids, `corpus_fingerprint` in `src/data/preflight.py`). Two runs are comparable only when it matches. It is in each policy's metric `meta` and in the pilot summary.
4. `scripts/check_index.py` re-runs the first retrieve for the locked 300 and compares the step-0 top ids with a trajectory file. Step 0 is used because it is raw BM25; the `retrieved` field on `rule_based` / `max_tools` has been through the lexical reranker. No GPU. Run it before spending GPU time.

**What has to be re-run.** One pilot on the v3 index, no training: `naive_rag`, `rule_based`, `max_tools` with `--artifact-suffix v3`. Then rebuild the sweep table and figure with `--frontier-tag v3`. The learned v3 exams on disk do not need to be re-scored; they are already on that index.

**Which preset.** Frozen policies ignore the reward scalar, so EM, F1, dollars, and the predictions are preset-independent; only the reward column moves. One pilot under `default` is enough for the EM-vs-$ frontier and the ceilings. A reward column next to λ=0 / 20 / 80 needs those trajectories rescored under each preset, which is a CPU rescore, not another Qwen pass.

**v2 stays as it is.** The untagged files, `frontier_sweep_table.json`, and `frontier_em_usd.png` remain the record of the v2 index. Ceilings 110 / 124 / 127 are v2-index numbers and keep that label until the v3 pilot lands.

## 2026-10-03 — v3 retrained on the z-scored observation: the policy branches

First sweep on the 16-d observation (2026-10-01 fix). Same 500 / 180 / locked 300, same 20 epochs, same validation pick, same corpus. **Only the feature scale changed.** That makes this a controlled comparison with the 2026-09-28 sweep, not a new experiment.

The 2026-09-28 `_v3` JSON and JSONL were **overwritten** by this run. Those numbers live in git at `3a2c83c`.

**Index.** Every artifact in this sweep reports `n_passages: 83120`, sha1 `18e880b5…`. That is the post-`build_train_valid` corpus, and the 2026-09-28 sweep used it too, so 15-d against 16-d is one index. It is **not** the 80,000 index behind `baseline_default.json` / `rule_based_default.json` / `max_tools_default.json`. Those are still unscored here, so the frozen-baseline comparison from 2026-10-01 is still open. `slice_meta.json` still says 80000; it was written 2026-08-27 and `build_train_valid.py` does not update it.

**Exams** (frozen argmax, locked 300, `_best.pt` picked on the 180):

| Preset | Selected epoch | EM | n_correct | $ | steps | retrieve | rewrite | verify | distinct sequences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frontier_lambda0` best | 11 | 0.333 | 100/300 | 3.11e-4 | 3.97 | 1.00 | 0.00 | 1.97 | **2** |
| `frontier_lambda0` last | 20 | 0.333 | 100/300 | 1.73e-4 | 2.02 | 1.02 | 0.00 | 0.00 | **2** |
| `frontier_lambda20` best = last | 20 | **0.343** | **103/300** | 5.31e-4 | 7.49 | 2.84 | 1.74 | 1.90 | **16** |
| `frontier_act02` best | 3 | 0.333 | 100/300 | 1.72e-4 | 2.00 | 1.00 | 0.00 | 0.00 | 1 |

**The headline: λ=20 emits 16 different action sequences on 300 questions.** Every previous sweep in this project emitted one. The step counts are no longer integers, which is the same fact read off the summary.

**It branches on the retrieval score.** Taking the decision-time observation (`observation_before.mean_score`, the standardized channel 0) at the step where the policy first has a choice:

| Preset | Decision | Branch A | Branch B |
|---|---|---|---|
| λ=20 | step 1 | `rewrite` (n=177): mean 20.4–46.4, median **35.5** | `retrieve` (n=123): mean 34.1–108.9, median **54.2** |
| λ=0 | step 1 | `verify` (n=295): median **41.3** | `stop` (n=5): 89.7–108.9, median **99.9** |

A single threshold on the mean score at **45.7** reproduces 278 of 300 λ=20 decisions (92.7%). For λ=0, a threshold near **90** picks out all 5 early stops with 2 false positives. Weak retrieval gets a rewrite; strong retrieval gets another retrieve, or on λ=0 an immediate stop. This is the behaviour the branching claim needed, and under `tanh(score / 5)` it was not expressible: that feature was 1.0 on all 300.

**Verify is still not paying for itself.** λ=0's selected recipe spends two verifies per question and changes **0 of 300 predictions** and **0 of 300 retrieved sets** against this run's retrieve→stop. It costs 3.11e-4 against 1.72e-4, about **1.8×** for nothing. The verify result never feeds back into retrieval in that recipe, so it cannot change an answer. λ=0 last epoch drops verify entirely and keeps the same 100/300 at the naive price.

λ=20 does act on the label, weakly. Share of post-verify steps that go back to retrieval: contradiction **8/15 (53%)**, support **124/452 (27%)**, neutral **13/104 (13%)**. Read that as an association, not a mechanism: the label correlates with the retrieval score, which is the feature we know the policy reads.

**EM.** Against this run's retrieve→stop (`frontier_act02` v3, same index), λ=20 is **+8 / −5** (Hotpot 0/−1, NQ +8/−4), 103 vs 100, and 268/300 predictions are identical. The union is 108/300. λ=0 best is 100/300 with identical predictions on 300/300.

So branching bought +3 EM for about 3× the dollars. That is a real controller and a bad trade at this λ. The 24-point oracle headroom from 2026-09-26 is still mostly unclaimed.

**Against the 15-d sweep** (same index, same slices, only the feature scale differs):

| | 15-d `tanh` (2026-09-28) | 16-d z-score (this run) |
|---|---|---|
| λ=0 best | EM 0.343 (103/300), 6.0 steps, 3.0 retrieve, 0 verify, **1 sequence** | EM 0.333 (100/300), 3.97 steps, 1.0 retrieve, 1.97 verify, **2 sequences** |
| λ=20 best | EM 0.333 (100/300), retrieve→stop, **1 sequence** | EM 0.343 (103/300), 7.49 steps, **16 sequences** |
| λ=80 best | EM 0.333 (100/300), retrieve→stop | EM 0.333 (100/300), retrieve→stop |

Overall EM did not move: 103 and 100 swapped presets. What changed is that the policy now conditions on the observation. Do not sell this as an accuracy result. One run per preset; the EM difference is inside the noise band 2026-09-26 already described (fixed recipes span about 5 points).

**λ=80 is unchanged**, retrieve→stop on all 20 greedy epochs, best at epoch 3. The arithmetic from 2026-09-12 still holds: at λ=80 extra dollars lose regardless of what the policy can see. A richer observation does not rescue a reward that makes tools −EV.

**Validation.** λ=0 best epoch 11 (valid reward 0.59548), λ=20 best epoch 20 (0.60218, so best = last), λ=80 best epoch 3 (0.56029). Greedy step counts on the 180 are non-integer on 10/20 epochs for λ=0 and **15/20** for λ=20, so the branching shows up during training, not only on the exam.

**Still open.** `naive_rag` / `rule_based` / `max_tools` on the 83,120 index, then `frontier_sweep_table_v3.json` and the rebuilt ceilings. Until that lands, no table may put a row from this sweep beside the frozen rows, and 110 / 124 / 127 stay labelled v2-index.

Sources: `learned_frontier_lambda0_v3.json` / `_v3last.json`, `learned_frontier_lambda20_v3.json` / `_v3last.json`, `learned_frontier_act02_v3.json` / `_v3last.json`, `train_policy_curve_frontier_v3_*.json`, and the matching JSONL.

## 2026-10-03 — Three seeds for the final run: plan and budget

Every learned number in this project is **one training run at seed 42**. The paper said so in half a sentence; it now says what that costs the reader (`paper/paper.tex`, Limitations). This entry is the plan to fix it, not a result.

**What the seed does and does not touch.** `experiment.seed` feeds `set_seed`, the env, the policy initialization, and the per-epoch shuffle. The frozen controllers are deterministic — temperature 0, BM25, rule-based branching — so `naive_rag` / `rule_based` / `max_tools` do **not** need re-running per seed. Nor do the exams: a frozen argmax pass over the locked 300 is deterministic given a checkpoint. **Seeds multiply training only**, which is 95% of the bill. That is the one piece of good news here.

**Why three.** Learned rows in the paper span 100 to 104 correct of 300. One run cannot say whether a four-answer difference is the objective or the draw. Three runs will not give a usable standard deviation either — with n=3 it is noise — so the reporting rule is **print all three values, or the min–max range**, never mean ± std. Three is enough to tell "the λ=20 policy branches on every seed" from "it branched once"; that is the claim we actually need to defend, and it is qualitative.

**Budget**, derived from this run's measured latency (`mean_total_latency_ms / mean_n_steps`) against the v3 shape of 500 train, 180 valid, 20 epochs, 2 exams:

| Preset | ms/step | train | valid-greedy | exams | one seed |
|---|---:|---:|---:|---:|---:|
| `frontier_lambda0` | 431 | 5.1 h | 1.7 h | 0.3 h | **7.1 h** |
| `frontier_lambda20` | 533 | 7.2 h | 3.9 h | 0.7 h | **11.8 h** |
| `frontier_act02` | 570 | 3.3 h | 1.1 h | 0.2 h | **4.7 h** |

One seed across all three is **23.5 GPU-h**; three seeds is **70.6 GPU-h**. Convert at your tier's units per hour — that rate changes often enough not to be written down here. Treat these as estimates: they extrapolate per-step latency measured at exam time to training rollouts, and they assume the same GPU.

**Tiers, in the order to give things up.**

1. **Full, 70.6 GPU-h.** Three seeds × three presets. Do this if units allow.
2. **Drop λ=80, 56.6 GPU-h.** `frontier_act02` is retrieve→stop on all 20 greedy epochs and the claim rests on the λ-arithmetic from 2026-09-12, not on a close call. Seeds buy the least here.
3. **λ=20 only, 35.4 GPU-h.** The branching result is the headline and the one a reviewer will press on. If only one preset gets seeds, it is this one.

Do **not** economize by evaluating greedy less often. The per-epoch valid pass is 29% of the bill (33% on λ=20) and halving it would be the obvious cut, but `_best.pt` is selected from exactly that curve. Changing its frequency changes the selection rule and breaks comparability with every run on disk.

**Enabling change.** `scripts/train_policy.py` had no `--seed`; the seed was reachable only by editing the config, so a sweep meant three config files. It now takes `--seed`, written back into `cfg["experiment"]["seed"]` rather than a local, so the override also reaches the stratified `--limit` draw and the env. Each seed needs its own `--checkpoint` and `--curve` or the runs overwrite each other; the recipe is `HOW_TO_RUN.md` §4.11.

**Order of work.** The frozen baselines on the 83,120 index (2026-10-01, still open) come **first**. They are a single pilot, they are seed-independent, and without them no learned row can be tabled beside a frozen one no matter how many seeds it has.

## 2026-10-05 — v4 is a second seed-42 draw, not the three-seed plan

Same config, same 83,120 index (sha1 `18e880b5…`), same seed, new filenames so `_v3` stays. Notebook: `notebooks/Frontier_Cost_Pressure_CA_RL_ARAG_v4.ipynb`. This is the comparison the 2026-10-03 entry said one run cannot support. It is still seed 42, run twice. The two policies do not match, so that seed did not pin the controller.

**Exams** (frozen argmax, locked 300):

| Preset | Selected epoch | EM | n_correct | $ | steps | retrieve | rewrite | verify | distinct sequences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frontier_lambda0` best = last | 20 | **0.347** | **104/300** | 5.37e-4 | 7.65 | 2.95 | 1.70 | 2.00 | **15** |
| `frontier_lambda20` best | 16 | **0.347** | **104/300** | 4.14e-4 | 5.89 | 2.94 | 1.95 | 0.00 | **6** |
| `frontier_lambda20` last | 20 | 0.333 | 100/300 | 4.85e-4 | 6.93 | 2.86 | 1.21 | 1.86 | **11** |
| `frontier_act02` best | 3 | 0.333 | 100/300 | 1.72e-4 | 2.00 | 1.00 | 0.00 | 0.00 | 1 |

**What repeated.** λ=80 is retrieve→stop on all 20 greedy epochs, and the exam matches the 2026-10-03 `act02` file on 300/300 predictions. Selected λ=0 and λ=20 beat this run's retrieve→stop by +4 EM (+9 / −5, union 109/300). The third retrieve copies the second (292/292 on λ=0, 290/291 on λ=20). Verify on the selected λ=0 policy still does not send a contradiction back to retrieval (0/28).

**What did not repeat.** The 2026-10-03 headline was λ=20 with 16 sequences and a rewrite-vs-retrieve split at mean score 45.7 (278/300). This draw's λ=20 is one recipe on 270/300: `retrieve, rewrite, rewrite, retrieve, retrieve, stop`. That recipe holds the entire +4. The other 30 questions are net 0 against retrieve→stop. The first action is rewrite on 272/300, including scores up to 97.9. The score mainly marks 8 early stops (85.4–108.9). λ=0 is the preset that branches here (15 sequences). Its first decision is rewrite below mean score 45.4 and verify above it (293/300), and those two verifies change 0 answers relative to λ=20 (same predictions on 300/300, retrieved sets on 294/300) at 5.37e-4 versus 4.14e-4.

**Accuracy.** Best EM is 104 here and 103 on 2026-10-03. v4 λ=20 matches that file on 295/300 (+1 / −0). One answer. Report the failure to repeat the controller, not a new EM. Seeds 43 and 44 are still unrun. Frozen baselines on 83,120 are still unrun, so no `_v4` row sits beside naive / rule / max_tools.

Sources: `learned_frontier_{lambda0,lambda20,act02}_v4.json` / `_v4last.json`, `train_policy_curve_frontier_v4_*.json`, and the matching JSONL.

## Observations template

| Date | Experiment | Observation | Implication |
|---|---|---|---|
| 2026-08-24 | Leaked-NQ 80k, `rule_based` (`2417c43`) | Hotpot verify 14 / 34 / 102; NQ support 150/150. After every verify, `stop`. | Verify was dead on leaked NQ. Frozen policy never used Hotpot contradictions. |
| 2026-08-24 | NQ corpus inspection (`prepare_data.py` anchors) | Each NQ gold was `{question} The answer is {gold}` twice. Recall@1 / Q_ground / P_hall on NQ were leakage artifacts. | **Implemented:** `--hf` uses Tevatron/wikipedia-nq (fallback TriviaQA/SQuAD). Preflight rejects leftover anchors. |
| 2026-08-24 | SQuAD fallback 80k Qwen 300-eval (`e8a4423`) | Overall EM 0.68 → 0.40. SQuAD 60/63/60, R@5 0.633. Verify SQuAD 0/24/126. Reward still naive > rule > max. | Single-hop is a ranking split. Extra tools still lose on λ. Re-run ablation on this slice. |
| 2026-08-27 | Tevatron NQ 80k Qwen 300-eval (`d456d26`) | Overall EM 0.33. NQ 41/41/44, R@5 0.587. Verify NQ 0/18/132. Ablation regenerated (EM 0.34). Reward naive > rule > max. | Intended NQ ranking snapshot. Distinct golds 847, not 7. Extra tools still lose on λ. |
| 2026-09-04 | Same slice, `calibration_score` fix | Lazy abstain no longer +0.6. Reward 0.580 / 0.531 / 0.509. Q_cal −0.137 / −0.147 / −0.128. Ablation default 0.499. | Train against the fixed calibration rule. Frozen ranking unchanged. |
| 2026-09-09 | First REINFORCE train, n=100 | Reward 0.649 → 0.564 → 0.643. Verify 0.39–0.55. Steps ~4. | Train samples tools. Not a ranking number. |
| 2026-09-10 | Learned 300-eval, frozen argmax | Retrieve→stop 300/300. EM 0.333, 59+41 correct, reward 0.580. Identical to naive. | Eval mode is naive. Verify-on-contradiction win condition did not fire. |
| 2026-09-12 | Reward arithmetic on committed 300-eval | Extra tools: +0.10 \(P_{\mathrm{act}}\), +0.001 \(\lambda\$\), +0.028 quality. Reward −0.071. Cost mix ~99% action tax / ~1% dollars. | Collapse to naive is the reward, not the MLP. \(P_{\mathrm{act}}\) is backwards for a dollar-aware paper. |
| 2026-09-13 | Free-cost trains + 300-eval | `correctness_only` argmax: 8 steps / 3 retrieve / 2 verify, EM 0.340 (102/300). `lambda_zero` argmax: retrieve→stop 300/300. | Trainer can learn tools. \(P_{\mathrm{act}}\), not λ, pins the ranking row to naive. |
| 2026-09-13 | λ=80 `act_penalty` sweep | All three learned evals retrieve→stop 300/300, identical to naive. Train shrank toward 2 steps. | Extra $ at λ=80 is still −EV even at \(P_{\mathrm{act}}=0\). Next: lower λ. |
| 2026-09-15 | Frontier retarget (config + trainer) | Presets λ=0 / 20 / 80; 40 epochs; greedy `eval_reward` per epoch. | Sweep now crosses break-even; sample-vs-greedy gap is logged. |
| 2026-09-16 | Colab `.pt` sync | λ=80 family checkpoints now local. Still retrieve→stop 300/300. No `lambda0` / `lambda20` JSON yet. | Do not cite the 09-15/16 v2 notebook outputs as the new family. |
| 2026-09-23 | λ=0 / 20 40-epoch train + 300-eval | λ=0: 6 steps / 2 rewrite / 2 verify, EM 0.347 (104/300). λ=20: 4 steps / 3 retrieve / 0 verify, EM 0.333 (100/300). Neither is retrieve→stop. | Last-epoch paths only. Revised 2026-09-26: best greedy epoch is naive for both. |
| 2026-09-24 | λ=80 `act02` 40-epoch exam + `--frontier` | Retrieve→stop 300/300, EM 0.333. Greedy was 2-step all 40 epochs. Sweep table + `frontier_em_usd.png` regenerated. | λ=80 matches naive on every epoch. The λ=0 / λ=20 points on that figure are last checkpoints. See 2026-09-26. |
| 2026-09-26 | Re-read of the λ=0 / 20 / 80 curves and 300 trajectories | Best greedy: λ=0 epoch 25 and λ=20 epoch 15, both retrieve→stop. Last epoch: 6-step recipe (train EM 0.40, eval 104/300) and 3 identical retrieves (answers = naive, reward gap ~0.0027). Each policy is one action sequence on 300/300. Verify contradiction 14 and neutral 55 never change the next action. Per-question oracle: naive∪λ=0 = 124; +max_tools = 127; +rule + `correctness_only` = 130. Naive-on-Hotpot + λ=0-on-NQ = 110 at ~1/3 of max_tools cost. | Fixed recipes are a flat frontier (~5 points). An observation-reading controller has 24 points of headroom (100 → 124). Do not plot last-epoch λ=0 / λ=20 as learned points. **Partly resolved 2026-10-03:** once the saturated score channels were fixed the policy does read the observation (λ=20, 16 sequences), but it claims only 3 of the 24 points. |
| 2026-09-26 | v3 trainer (`frontier_v3.yaml`, notebook `_v3`) | 500 train / 180 valid from the HF train split. Advantage is sampled minus cached naive. Observation is 15-d (BM25 gap + verify one-hot, no dataset id). 20 epochs. Artifacts use a `v3` suffix. | Design lock for the next sweep. Scored 2026-09-28. The v2 last-epoch files stay the record of the earlier recipes. |
| 2026-09-28 | v3 λ=0 / 20 / 80 exams, 20 epochs, valid-greedy `_best.pt` | Selected λ=0 is epoch 11: one 6-step recipe, EM 0.343 (103/300, Hotpot 60 / NQ 43), $ 4.21e-4. Second retrieve changes top-5 on 106/300; third retrieve copies it on 300/300. λ=0 last matches retrieve→stop answers after two rewrites ($ 3.21e-4). λ=20 epoch 7 and λ=80 all 20 greedy epochs are retrieve→stop, EM 0.333 (100/300). Within-run union of the recipe and retrieve→stop is 109/300. Greedy verify is 0. | Validation pick moved λ=0 off retrieve→stop by +3 EM. The policy is still one sequence. The verify one-hot is unused on the exam. |
| 2026-10-01 | Index audit of the v3 exams vs the frozen rows | v3 train/valid golds were merged into `corpus.jsonl`, so BM25 changed for the locked 300: 20/300 different top ids, retrieve→stop is Hotpot 60 / NQ 40 against the ranking row 59 / 41. The frontier figure, the sweep table, and the 110 / 124 / 127 ceilings were still built from old-index files even under `--frontier-tag v3`. | One index per table. `--artifact-suffix` forks every policy; `--frontier-tag` refuses to fall back to untagged files; every artifact carries a corpus fingerprint; `check_index.py` verifies before a GPU run. Re-run naive / rule / max on the v3 index and rebuild the ceilings. |
| 2026-10-01 | Score-channel scale (`tanh(score/5)` vs train-slice z-score) | v3 λ=0 top-1 is 26.2–122.3 (median 49.3). `tanh(x/5)` is 0.9999 or 1.0 on all 300, and the mean channel is too. Gap was the only score input that moved. Mean `< 3` never fires: raw means stay above 20, reranked means above 7. | Z-score mean, top-1, gap, and top-5 min with the 100-question train slice and clip to [−3, 3]. Vector is 16-d. Old checkpoints do not load. The `< 3` abstain bonus stays in the code and stays dead until a reward change retargets it. |
| 2026-10-03 | v3 retrained on the 16-d z-scored observation (same config, same 83,120 index; only the feature scale changed) | λ=20 emits **16 distinct action sequences** on 300 questions, where every earlier sweep emitted one; a single threshold on the decision-time mean BM25 at 45.7 reproduces 278/300 of its first decisions. EM 0.343 (103/300) at 5.31e-4 vs this run's retrieve→stop 0.333 (100/300) at 1.72e-4 (+8 / −5, union 108/300). λ=0 stops early on exactly the 5 strongest-retrieval questions; its two verifies change 0/300 predictions at 1.8× the cost. λ=80 unchanged. Overwrote the 2026-09-28 `_v3` files (git `3a2c83c`). | The 2026-09-26 win condition is met: the controller reads the observation. Report behaviour, not accuracy — best EM is 103 under both observations and only changed preset. Verify still needs to gate a re-retrieve to earn its cost. Frozen baselines are **still** on the 80k index, so no `_v3` row may sit beside them. |
| 2026-10-05 | v4, second seed-42 training run of the same 16-d config (new filenames; `_v3` kept) | Selected λ=0 (epoch 20) and λ=20 (epoch 16) tie at EM 0.347 (104/300) with identical predictions. λ=20 is one 6-step recipe on 270/300 and that recipe holds the +4 vs retrieve→stop (+9 / −5, union 109) at 4.14e-4. λ=0 uses 15 sequences; a threshold at 45.4 separates rewrite vs verify on 293/300, and the two verifies change 0 answers (5.37e-4). λ=20 last copies the first retrieve and matches retrieve→stop. λ=80 unchanged, predictions identical to the 2026-10-03 `act02` file. | The 16-sequence λ=20 controller did not repeat. 104 vs 103 is one answer. Do not move the headline from behaviour to accuracy. This is not seeds 43 and 44. Frozen baselines are still the 80k index, so no `_v4` row may sit beside them. |
