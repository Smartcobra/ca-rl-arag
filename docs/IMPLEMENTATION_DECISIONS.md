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

**Run this note describes:** leaked-NQ 80k Qwen ranking pilot, commit `2417c43` (2026-08-23). Superseded later the same day by `e8a4423` (SQuAD). Historical trajectories: `results/trajectories/rule_based_default.jsonl` at that commit.

Trajectory counts (150 Hotpot + 150 NQ, leaked anchors):

| Split | contradiction | neutral | support |
|---|---:|---:|---:|
| HotpotQA | 14 | 34 | 102 |
| Natural Questions | 0 | 0 | 150 |

Junior reading: `verify` is a “does the evidence agree with this answer?” check. On leaked NQ it always said yes (answer-anchor ceiling). On Hotpot it mixed — 14 contradictions, 34 neutrals — so the signal discriminates on the hard split.

`rule_based` still ignores it for search. After every `verify` the next action was `stop` (300/300). Full write-up of the **current** (SQuAD) counts: `docs/RESULTS.md` §8.

## 2026-08-24 — NQ answer-anchors are label leakage; replaced with DPR Wikipedia (SQuAD fallback)

**Status:** implemented in `scripts/prepare_data.py` / `src/data/wiki_passages.py`. Ranking metrics in `docs/RESULTS.md` (`e8a4423`) describe the **SQuAD fallback** run. Do not mix those numbers with leaked-anchor NQ (`2417c43`).

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

Implication: both splits now rank. Milestone 3 can condition on verify. Preferred future single-hop source is still Tevatron NQ if the download fits.

## Observations template

| Date | Experiment | Observation | Implication |
|---|---|---|---|
| 2026-08-24 | Leaked-NQ 80k, `rule_based` (`2417c43`) | Hotpot verify 14 / 34 / 102; NQ support 150/150. After every verify, `stop`. | Verify was dead on leaked NQ. Frozen policy never used Hotpot contradictions. |
| 2026-08-24 | NQ corpus inspection (`prepare_data.py` anchors) | Each NQ gold was `{question} The answer is {gold}` twice. Recall@1 / Q_ground / P_hall on NQ were leakage artifacts. | **Implemented:** `--hf` uses Tevatron/wikipedia-nq (fallback TriviaQA/SQuAD). Preflight rejects leftover anchors. |
| 2026-08-24 | SQuAD fallback 80k Qwen 300-eval (`e8a4423`) | Overall EM 0.68 → 0.40. SQuAD 60/63/60, R@5 0.633. Verify SQuAD 0/24/126. Reward still naive > rule > max. | Single-hop is a ranking split. Extra tools still lose on λ. Re-run ablation on this slice. |
