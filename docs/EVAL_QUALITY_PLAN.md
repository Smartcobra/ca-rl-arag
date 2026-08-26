# Eval Quality Plan

**Status (2026-08-24):** the locked 300-eval ranking run exists. Current numbers live in [`RESULTS.md`](RESULTS.md) (80k Qwen, commit `e8a4423`: 150 Hotpot + 150 **SQuAD**). Per-dataset reporting, stratified `--limit`, ABSTAIN parsing, 50k corpus preflight, and no-anchor NQ/SQuAD golds are in the tree. Remaining gap: `reward_ablation_table.json` is still the leaked-NQ stratified 100 (`2417c43`) — re-run `python scripts/run_reward_ablation.py` on the SQuAD files. Preferred single-hop source remains Tevatron NQ if the download fits.

**This file is a hygiene plan, not a ranking table.** The diagnosis below describes the 40-example Qwen mix (~2026-08-20), before the locked 300-eval. Keep it as history; do not cite those 40-ex numbers as current.

Plan for fixing eval reporting, ABSTAIN parsing, slice bias, stale ablation artifacts, and repo hygiene. **No ranking conclusions until this is done.**

The 40-example headline numbers mix two regimes, and several `results/` artifacts still describe a different experiment than the Qwen run.

---

## Diagnosis

- **NQ is saturated** (EM 0.933, identical across policies). That is expected given answer-anchor passages in `scripts/prepare_data.py` (`The answer is {ans}`). NQ currently cannot rank policies.
- **HotpotQA is the only discriminative split**, and there the gaps are 1–2 EM hits on 25 examples plus a ~50% abstain rate. At that size, cost (λ) decides reward ranking, not quality.
- **`--limit 40` is not a random 40.** `eval_slice.jsonl` is `hotpot + nq` (`prepare_data.py`). Prefix slicing keeps all 25 Hotpot and 15/25 NQ. Ablation `--limit 30` is worse: 25 Hotpot + 5 NQ.
- **ABSTAIN leak is a parser bug, not a policy effect.** `HuggingFaceGenerator.generate` only looks at `answer.upper()[:20]`, and even then only abstains if the string *starts with* `ABSTAIN` or evidence is empty. Downstream, `rag_baseline` / `agentic_rag` treat abstain as `answer.upper() == "ABSTAIN"`. Mixed strings like `"… ABSTAIN"` stay answers and pollute EM/F1, grounding, and calibration.
- **`results/` is two experiments glued together.** `pilot_summary_default.json` is Qwen (EM ~0.45 / 40). `reward_ablation_table.json` is extractive (EM 0.033). `run_pilot.py` regenerates `reward_ablation.png` from that stale table.
- **Docs vs git:** `docs/` exists locally (including `IMPLEMENTATION_DECISIONS.md`) but is **untracked**. On the branch, README links are broken. 22 `*.pyc` files are tracked; `.gitignore` is four copies of `.env`.

Do not treat naive > agentic on reward as a finding until the parser is fixed, the mix is honest, metrics are per-dataset, and eval is ~200–300.

---

## Recommended order

Parser fix and per-dataset reporting first, then slice/limit, then GPU reruns, then docs/README refresh. Rerunning before the parser fix would bake the leak into the new numbers.

```text
1. Hygiene (gitignore, untrack pyc, add docs/)     — no GPU
2. ABSTAIN parse + prompt                           — no GPU, unit tests
3. Per-dataset aggregate in evaluate/summary/plots  — no GPU
4. Stratified / no-prefix limit in pilot + ablation — no GPU
5. Grow eval slice in config + re-prepare --hf      — download only
6. Re-run Qwen pilot + ablation + plots             — GPU
7. Refresh README / RESULTS / EXPERIMENT_LOG        — after numbers exist
```

---

## 1. Always report per-dataset metrics

**Where:** `src/metrics.py`, `src/evaluate.py`, `scripts/run_pilot.py`, `scripts/run_reward_ablation.py`, then `scripts/plot_results.py`.

Trajectory rows already have `"dataset"`. Aggregation does not use it.

**Change `aggregate_metrics` / evaluate summary to emit:**

- Top-level overall (current fields)
- `by_dataset`: `{hotpot_qa: {...}, natural_questions: {...}}` with the same means
- Extra counts that the 40-example analysis needed: `n_abstained`, `abstain_rate`, `n_correct`, `n_examples`

`evaluate_baseline` / `evaluate_agent` should compute overall + per-dataset from the same rows (one pass, group by `row["dataset"]`).

**Write it through to artifacts:**

- `pilot_summary_*.json`: `results[policy].by_dataset` plus a top-level `n_examples_by_dataset`
- Console print: overall line, then Hotpot line, then NQ line
- Ablation table: either nest `by_dataset` per preset or write a sibling `reward_ablation_by_dataset.json` so plots/docs do not go back to a single EM

**Plots (after the next Qwen run):** grouped bars for EM/F1/reward/abstain-rate **by dataset**, not only overall. Overall bars stay, but they must not be the only figure.

**Rule for write-ups:** every table in README / RESULTS / EXPERIMENT_LOG is overall + Hotpot + NQ. If NQ is flat, say so in one sentence (saturated / answer-anchor corpus). Ranking language is Hotpot-only until NQ has a real index.

---

## 2. Fix ABSTAIN parsing and the prompt

**Where:** `src/generation/llm.py` (`HuggingFaceGenerator.generate`). Optionally a tiny shared helper used by baseline/agent/env as a second line of defense.

**Parser (single helper, e.g. `_parse_answer_or_abstain`):**

1. Take first non-empty line, strip `Answer:` / `Final answer:` prefixes (already done).
2. If the whole string is `ABSTAIN` (case-insensitive, optional punctuation) → abstain.
3. If it **ends with** an `ABSTAIN` token (`… ABSTAIN`, `… ABSTAIN.`, `…\nABSTAIN`) → strip the token.
   - If anything remains → treat as **answer** (model answered, then hedged).
   - If nothing remains → abstain.
4. If it **starts with** `ABSTAIN` then extra text → abstain (do not keep the tail as an answer).
5. Drop the `[:20]` substring check. It both misses trailing leaks and inconsistently keeps mixed strings when evidence exists.
6. Return `("ABSTAIN", "abstain")` or `(clean_span, "answer")` — never a mixed string.

Tighten `rag_baseline.py` / `agentic_rag.py` / `rag_env.py` abstain flags to use `mode == "abstain"` (or the helper), not `answer.upper() == "ABSTAIN"` alone.

**Prompt:** exclusive choice, not an optional suffix.

- Reply with **either** a short answer span **or** the single token `ABSTAIN`.
- Never append `ABSTAIN` after an answer. Never answer after `ABSTAIN`.
- No explanation, no quotes.

**Tests (add a small unit file; `smoke_test.py` does not cover this):**

| Input | Expected |
|---|---|
| `ABSTAIN` | abstain |
| `Secretary of State for Constitutional Affairs ABSTAIN` | answer, no trailing token |
| `ABSTAIN: not enough evidence` | abstain |
| `Answer: Paris` | `Paris` |
| empty | abstain if `allow_abstain` |

Do this **before** any Qwen re-run. 2–4 leaks per policy on 40 examples is enough to move Hotpot EM by the same 1–2 hits that currently “rank” policies.

---

## 3. Stop `--limit` from rewriting the dataset mix

**Root cause:** `examples = examples[: args.limit]` in `run_pilot.py` and `examples[: args.limit]` in `run_reward_ablation.py`, on a hotpot-first JSONL.

**Immediate (50-example slice):**

- Default pilot: **no prefix limit** (full eval file, currently 50). Change docs/README that still say `--limit 40` / `--limit 30`.
- Ablation: same mix as the pilot, not 30.

**If a debug `--limit` stays:** make it **stratified**, not a prefix. With seed 42, take `round(limit * n_ds / n_total)` from each dataset (and document leftover rounding). Optional `--shuffle` of the eval file is not enough by itself: a shuffle then prefix can still skew; stratified is the honest `--limit`.

**Do not use `--limit 40` in HOW_TO_RUN examples anymore.** Use full eval, or `--limit 50` with a comment that it must preserve the config mix.

---

## 4. Grow eval to ~200–300 before ranking

**Config (`configs/default.yaml`, locked):** `eval_hotpot: 150`, `eval_nq: 150` → 300.

**Proposed slice (pick one and lock it in config + `slice_meta.json`):**

| Split | Hotpot | NQ | Total | Rationale |
|---|---:|---:|---:|---|
| A (balanced) | 125 | 125 | 250 | Mix matches the current 50/50 story |
| B (signal-weighted) | 200 | 50–100 | 250–300 | Extra budget on the only non-saturated set |

Recommendation: **A for the public table** (honest mix), and always read Hotpot as the ranking split. NQ at 125 will still be near-ceiling until answer-anchors go away; growing NQ does not buy policy signal.

**Prep:** bump `eval_*` (train can stay 100 for now), `python scripts/prepare_data.py --hf`. Corpus will grow (more Hotpot distractor contexts). NQ still gets anchors — log that in `IMPLEMENTATION_DECISIONS.md` as a known ceiling, not a model success.

**Compute (order of magnitude):** current pilot is 40 × 3 policies. 300 × 3 is ~7.5×. Ablation is 6 presets × 1 policy × N examples. If GPU time is tight: full 300 for the **pilot**; ablation on a **stratified 50 or 100** from the same slice, labeled as such. Do not mix extractive ablation with Qwen pilot in `results/`.

**After the run:** report Hotpot CIs or at least raw `n_correct / n` (e.g. 40/125 vs 50/125). Still no “policy A wins” language if the EM gap is a handful of items unless a McNemar / bootstrap is added (optional, later).

---

## 5. Refresh stale `results/` and README from one Qwen setup

**Stale now:**

- `results/metrics/reward_ablation_table.json` and `ablation_rule_based_*.json` (EM 0.033)
- `results/figs/reward_ablation.png` (plotted from that table)
- README “Pilot results” table (extractive 0.050 / 0.025 / 0.075)
- `docs/RESULTS.md` §3 and `docs/EXPERIMENT_LOG.md` HF table (same legacy numbers)

**Qwen pilot JSON is already in `pilot_summary_default.json` (EM ~0.45) but it is the biased 40-mix and pre-parser-fix — do not promote it.**

**Sequence after sections 1–4:**

```bash
python scripts/prepare_data.py --hf          # new eval counts
python scripts/run_pilot.py --run-env-check  # no prefix --limit
python scripts/run_reward_ablation.py        # same mix / documented subset
python scripts/plot_results.py
```

Then replace README / RESULTS / EXPERIMENT_LOG tables with **overall + per-dataset**, note NQ saturation, and state that ranking is deferred until n=300. Keep one clearly labeled “legacy extractive, 40-ex, pipeline sanity only” appendix if you want history; do not leave it as the headline.

---

## 6. Repo hygiene

**`.gitignore`** — replace the four `.env` lines with a real ignore list:

```gitignore
.env
__pycache__/
*.pyc
*.pyo
.DS_Store
.idea/
```

**Untrack the 22 committed bytecode files** (they are on `feature_baseline`):

```bash
git rm -r --cached '**/__pycache__'
```

Working tree can keep running; they just stop being versioned. Also ignore the untracked `*.cpython-313.pyc` files so they do not show up as `??`.

**Docs:** they are written locally but **not on the branch**. Commit the existing `docs/` tree, including `docs/IMPLEMENTATION_DECISIONS.md`. That closes the “folder doesn’t exist on git” item. Add a dated section covering:

- NQ answer-anchor ceiling (why EM≈0.93 is not a policy result)
- Eval file order + why prefix `--limit` biases the mix
- ABSTAIN exclusive-or parsing
- Per-dataset reporting as a required eval contract
- Eval scale-up to 300 and “no ranking until then”
- Qwen vs extractive: `results/` must be one generator

Also add `.DS_Store` / `.idea/` while touching gitignore (both currently untracked).

---

## What “done” looks like

| Check | Pass condition |
|---|---|
| Parser | No prediction contains the substring `ABSTAIN` unless the whole answer is `ABSTAIN`; unit tests for the leak examples |
| Summary JSON | Every policy has `by_dataset.hotpot_qa` and the loaded single-hop key (`natural_questions` / `trivia_qa` / `squad`) with EM/F1/reward/abstain |
| `--limit` | `limit=40` on 150+150 (or 25+25) returns ~20+20, never 40 Hotpot + 0 NQ or 25+15 |
| Eval size | `slice_meta.json` eval total 300; mix 150 Hotpot + 150 single-hop. Current ranking snapshot: 150 SQuAD (`e8a4423`) |
| `results/` | Ablation EM in the same ballpark as the Qwen pilot (not 0.033 next to 0.45). **Open:** ablation JSON is still leaked-NQ EM 0.67 vs SQuAD ranking EM ~0.40 |
| README | Headline table is Qwen + per-dataset, or explicitly “not a ranking” |
| Git | 0 tracked `*.pyc`; `docs/IMPLEMENTATION_DECISIONS.md` is in the tree; `.gitignore` covers pycache |

---

## Out of scope for this pass

- Replacing NQ answer-anchors with DPR Wikipedia passages — **done** (`src/data/wiki_passages.py`). The committed ranking snapshot used the **SQuAD** fallback (`e8a4423`); do not cite leaked-anchor NQ EM as a policy result.
- Statistical tests / Pareto claims.
- Changing reward weights because naive “won” on the 40-mix.

First code PR: hygiene + ABSTAIN helper + per-dataset aggregate + stratified limit — then the GPU jobs.
