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

`rule_based` still ignores it for search. After every `verify` the next action was `stop` (300/300). Full write-up of the **current** (Tevatron NQ) counts: `docs/RESULTS.md` §8.

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

**Not chosen (yet):** GRPO/PPO, a learned critic, query/passage text in the policy, or registering `learned` in `run_pilot.py` / `get_policy()`.

**Data lock:** train on `train_slice.jsonl` only (60 Hotpot + 40 NQ = 100). The script has no `--split` flag and refuses any path whose filename contains `eval`. The locked 300-example eval is not opened during training. `--limit` is a stratified cap on **train**.

**Why tiny:** 100 trajectories cannot support a wide net. Hidden width is capped at 32 (`ValueError` above that). Observation is the existing env vector (including verify `support` / `contradiction`). Reward is the **fixed** 2026-09-04 calibration rule.

**Update:** per-episode REINFORCE with an EMA reward baseline and entropy 0.01. Sparse terminal reward from `AgenticRAGEnv`. Learning curve: `results/metrics/train_policy_curve.json` (mean reward per epoch). Checkpoints: `results/checkpoints/learned_policy.pt` and `_best.pt`.

**Eval:** ranking the checkpoint on the 300 is a later, separate job. Train mean reward is not a ranking number.

## Observations template

| Date | Experiment | Observation | Implication |
|---|---|---|---|
| 2026-08-24 | Leaked-NQ 80k, `rule_based` (`2417c43`) | Hotpot verify 14 / 34 / 102; NQ support 150/150. After every verify, `stop`. | Verify was dead on leaked NQ. Frozen policy never used Hotpot contradictions. |
| 2026-08-24 | NQ corpus inspection (`prepare_data.py` anchors) | Each NQ gold was `{question} The answer is {gold}` twice. Recall@1 / Q_ground / P_hall on NQ were leakage artifacts. | **Implemented:** `--hf` uses Tevatron/wikipedia-nq (fallback TriviaQA/SQuAD). Preflight rejects leftover anchors. |
| 2026-08-24 | SQuAD fallback 80k Qwen 300-eval (`e8a4423`) | Overall EM 0.68 → 0.40. SQuAD 60/63/60, R@5 0.633. Verify SQuAD 0/24/126. Reward still naive > rule > max. | Single-hop is a ranking split. Extra tools still lose on λ. Re-run ablation on this slice. |
| 2026-08-27 | Tevatron NQ 80k Qwen 300-eval (`d456d26`) | Overall EM 0.33. NQ 41/41/44, R@5 0.587. Verify NQ 0/18/132. Ablation regenerated (EM 0.34). Reward naive > rule > max. | Intended NQ ranking snapshot. Distinct golds 847, not 7. Extra tools still lose on λ. |
| 2026-09-04 | Same slice, `calibration_score` fix | Lazy abstain no longer +0.6. Reward 0.580 / 0.531 / 0.509. Q_cal −0.137 / −0.147 / −0.128. Ablation default 0.499. | Train against the fixed calibration rule. Frozen ranking unchanged. |
