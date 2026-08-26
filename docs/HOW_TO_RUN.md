# How to Run — Milestone 2 Pipeline Guide

This guide explains **why** each command exists, **which datasets** are used, **what evaluation metrics** mean, **how to run** each step, and **what output** you get.

Work from the project folder:

```bash
cd agentic_rag_rl
source ../.venv/bin/activate   # or your own venv
pip install -r requirements.txt   # includes torch + transformers for Qwen
```

Default generator is **local** `Qwen/Qwen2.5-3B-Instruct` (`generation.backend: huggingface`). Weights download once to `~/.cache/huggingface` and run on CUDA/MPS/CPU.

Recommended order (same as the quick start):

```bash
python scripts/smoke_test.py                  # forces extractive (offline); overwrites data/processed
python scripts/prepare_data.py --hf           # 150+150 questions + unused-Hotpot pool (~80k passages)
python scripts/run_pilot.py --run-env-check   # preflight: eval=300 AND corpus>=50k, then Qwen
python scripts/run_reward_ablation.py         # same preflight; stratified 100 from the same file
python scripts/plot_results.py                # metrics → results/figs/*.png
```

Fast extractive-only debug (not a ranking run; `extractive.yaml` has no 50k corpus floor):

```bash
python scripts/run_pilot.py --config configs/extractive.yaml --limit 50
```

If you point **default.yaml** at a synthetic or 2k-passage corpus, the ranking scripts exit before loading Qwen. Use `--skip-data-check` only for that debug path.

---

## 1. Why these four commands?

| Step | Command | Why run it |
|---|---|---|
| 1 | `smoke_test.py` | Fast offline sanity check. Confirms imports, synthetic data, baseline RAG, agent loop, Gymnasium env, and reward presets work **before** spending time on downloads or pilots. |
| 2 | `prepare_data.py` | Builds the train/eval question slices and BM25 corpus. Without this, pilot/ablation have nothing to evaluate. |
| 3 | `run_pilot.py` | Main Milestone-2 experiment: compare **naive RAG**, **rule-based agent**, and **max-tools agent**; optionally verify the RL env. Produces metrics + trajectory logs. |
| 4 | `run_reward_ablation.py` | Sweeps reward-weight presets on a fixed policy/slice so you can justify α/β/γ/λ choices (reviewer comment) and later write the paper ablation section. |

They are sequenced so you never debug data/reward issues on a broken pipeline.

---

## 2. Datasets used

Primary datasets follow **Scope Memo V2 §7**:

| Dataset | HuggingFace id (when `--hf`) | Role |
|---|---|---|
| **HotpotQA** (distractor) | `hotpotqa/hotpot_qa` | Multi-hop QA; supporting titles enable grounding metrics |
| **Natural Questions** (preferred) | `Tevatron/wikipedia-nq` (DPR Wikipedia 100-word passages) | Single-hop QA with real evidence; teaches when *not* to over-retrieve. |
| **SQuAD / TriviaQA** (fallback) | `rajpurkar/squad` or Tevatron TriviaQA/SQuAD | Same single-hop slot if Tevatron NQ is too heavy. **Committed ranking run `e8a4423` used SQuAD.** |

### Modes

| Flag | What you get |
|---|---|
| `--hf` | Real Hotpot + single-hop (DPR Wikipedia NQ if it loads; else TriviaQA / SQuAD), plus an unused-Hotpot **distractor pool** (target 80k, floor 50k). Locked eval stays 150+150. Golds are real passages, not `{question} The answer is {gold}`. Check `slice_meta.json` (`nq_corpus`, `n_nq_anchor`). |
| `--synthetic` | Offline closed fact corpus (capitals/scientists). Use when you have no network or want a perfect EM smoke/pilot. |
| *(no flag)* | Tries HuggingFace first; falls back to synthetic if download fails. |

### Default slice sizes (`configs/default.yaml`)

Locked **plan A** (balanced mix). After rebuilding with `--hf`, single-hop uses real passages (NQ / TriviaQA / SQuAD) and can rank stop vs over-retrieve. Historical RESULTS tables from the answer-anchor NQ corpus (`2417c43`) are a different run. Current committed metrics (`e8a4423`) are **150 Hotpot + 150 SQuAD**.

| Split | Hotpot | Single-hop (NQ or fallback) | Typical total |
|---|---|---|---|
| Train | 60 | 40 | 100 |
| Eval | 150 | 150 | **300** |

`--limit` is optional and **stratified** (not a JSONL prefix). Prefix `--limit 40` on this file would be 40 Hotpot + 0 NQ. Ablation defaults to a stratified 100 from the same 300 (labeled as a subset, not the ranking table).

Actual counts after prepare are written to `data/processed/slice_meta.json`. After `--hf` you should see `n_eval: 300`, `n_passages` ≥ 50,000 (usually 80,000), and **`n_nq_anchor`: 0**. The committed ranking snapshot has `nq_corpus: squad`. A ~2,276-passage file is the old tiny index — do not start a Qwen ranking run on it.

### Files written by data prep

| File | Contents |
|---|---|
| `data/processed/train_slice.jsonl` | Training questions |
| `data/processed/eval_slice.jsonl` | Evaluation questions |
| `data/processed/corpus.jsonl` | Shared passages for BM25 |
| `data/processed/slice_meta.json` | Counts, source (`huggingface_nq_hotpot` or `synthetic`), paths |

Each example has fields like: `id`, `dataset`, `question`, `answer` / `answers`, `supporting_titles` (Hotpot).

More detail: `docs/data_cards/hotpotqa.md`, `docs/data_cards/natural_questions.md`, `docs/data_cards/squad.md`.

---

## 3. Evaluation matrix (metrics)

The project reports **quality**, **grounding/safety**, **efficiency/cost**, and the **aggregate reward**. This matches Scope Memo V2 §9 (quality–cost trade-off is the headline).

**Always split HotpotQA and the single-hop set.** `aggregate_metrics` groups rows by `dataset` and emits the same means under `by_dataset` (`hotpot_qa`, `natural_questions`, `trivia_qa`, or `squad`). Overall is mix-weighted and is a headline only after you have seen the mix. Rebuild the corpus with `--hf` before RL; preflight rejects leftover NQ answer-anchors.

### Quality

| Metric | Meaning |
|---|---|
| **EM** (`mean_em`) | Exact Match after answer normalization (lowercasing, strip articles/punctuation) |
| **Token F1** (`mean_f1`) | Token overlap F1 between prediction and gold |
| **n_correct** | Number of examples with EM = 1 |
| **n_abstained / abstain_rate** | Count and fraction of abstaining examples |
| **by_dataset** | Same fields, grouped by `hotpot_qa` and the loaded single-hop id (`natural_questions` / `trivia_qa` / `squad`) |

### Grounding / calibration (also used inside reward)

| Metric | Meaning |
|---|---|
| **Q_ans** | `0.5 * EM + 0.5 * F1` |
| **Q_ground** | Claim–evidence support (NLI/lexical) + Hotpot gold-title recall when available |
| **Q_cal** | Calibration: justified abstain vs confident wrong |
| **P_hall** | Hallucination / unsupported-claim penalty |

### Efficiency / cost

| Metric | Meaning |
|---|---|
| **total_usd** | FinOps $ from the price card (retrieve, rewrite, rerank, verify, generate) |
| **total_tokens** | Prompt + completion tokens (virtualized for extractive backend) |
| **total_latency_ms** | Wall-clock tool latency |
| **n_steps** | Actions in the trajectory |
| **n_retrieve / n_rewrite / n_rerank / n_verify** | Action mix |
| **usd_per_correct** | Total $ / number of EM-correct answers (∞ if none correct) |

### Aggregate reward

\[
R = \alpha Q_{\mathrm{ans}} + \beta Q_{\mathrm{ground}} + \gamma Q_{\mathrm{cal}}
- \lambda(C_{\mathrm{tok}} + C_{\mathrm{ret}}) - \mu C_{\mathrm{lat}}
- P_{\mathrm{hall}} - P_{\mathrm{act}} - P_{\mathrm{bud}}
\]

Defaults and ablation presets: `configs/reward_weights.yaml`, `docs/REWARD_DESIGN.md`.

### Policies compared in the pilot

| Policy | Behavior |
|---|---|
| **naive_rag** | Retrieve once → generate/stop (Lewis-style baseline) |
| **rule_based** | Threshold policy over retrieve/rewrite/rerank/verify/stop. It *runs* verify, but on the current trajectories it does **not** re-retrieve or rewrite after a contradiction — it just stops. |
| **max_tools** | Uses tools up to caps (high-cost reference) |

**Verifier vs frozen policy (read this before RL).** Lexical NLI is not a dummy label. On the current SQuAD ranking run (`e8a4423`), Hotpot was 14 contradiction / 34 neutral / 102 support and SQuAD was 0 / 24 / 126. After every `verify` — including all 14 Hotpot contradictions and 24 SQuAD neutrals — `rule_based` just stops. Details: [`RESULTS.md` §8](RESULTS.md) and [`IMPLEMENTATION_DECISIONS.md`](IMPLEMENTATION_DECISIONS.md). The leaked-NQ 150/150-support table (`2417c43`) is historical.

---

## 4. Command-by-command: how to run and what you get

### 4.1 Smoke test

**Why:** Catch broken installs / API mismatches in seconds, no network.

```bash
python scripts/smoke_test.py
```

**What it does:**
- Builds a tiny synthetic corpus in memory / `data/processed/`
- Runs one baseline RAG example
- Runs one rule-based agentic episode
- Steps the Gymnasium env (`retrieve` → `stop`)
- Loads all reward ablation presets

**Console output (success):**
```text
SMOKE OK
baseline_pred: ... em= ... reward= ...
agent_pred: ... actions= [...] em= ...
```

**Artifacts:** May refresh `data/processed/*.jsonl` with synthetic data.  
If you already prepared HuggingFace data and want to keep it, re-run `prepare_data.py --hf` after smoke test.

---

### 4.2 Prepare data

**Why:** Create the NQ + Hotpot slices and corpus used by every later script.

```bash
# Real datasets (needs network + HuggingFace)
python scripts/prepare_data.py --hf

# Offline synthetic (no network)
python scripts/prepare_data.py --synthetic
```

**Optional:**
```bash
python scripts/prepare_data.py --config configs/default.yaml
```

**Console output (example for `--hf`; committed ranking run used the SQuAD fallback):**
```json
{
  "source": "huggingface_nq_hotpot",
  "seed": 42,
  "n_train": 100,
  "n_eval": 300,
  "n_passages": 80000,
  "n_hotpot_distractor": 77898,
  "n_nq_wiki": 16,
  "n_nq_anchor": 0,
  "train_by_dataset": {"hotpot_qa": 60, "squad": 40},
  "eval_by_dataset": {"hotpot_qa": 150, "squad": 150},
  "nq_corpus": "squad",
  "nq_hf_dataset": "rajpurkar/squad",
  "retrieval_diag": {
    "by_dataset": {
      "hotpot_qa": {"recall@5": 0.927},
      "squad": {"recall@5": 0.633}
    }
  },
  "paths": { ... }
}
Wrote processed slices to .../data/processed
```

If Tevatron NQ loads, `nq_corpus` is `dpr_wikipedia_w100` and `dataset` is `natural_questions`. Always read `slice_meta.json` rather than assuming NQ.

Optional: `python scripts/prepare_data.py --hf --distractor-pool 80000` (0 disables the pool). `python scripts/retrieval_diagnostics.py` reprints BM25 gold recall@k.

**Artifacts:**

| Path | Role |
|---|---|
| `data/processed/train_slice.jsonl` | Train questions |
| `data/processed/eval_slice.jsonl` | Eval questions |
| `data/processed/corpus.jsonl` | Passages |
| `data/processed/slice_meta.json` | Metadata summary |

---

### 4.3 Run pilot

**Why:** End-to-end comparison of baselines on the eval slice; also proves the RL environment can roll episodes.

```bash
python scripts/run_pilot.py --run-env-check
```

**Preflight (before Qwen loads).** With `configs/default.yaml` this is **not** `--run-env-check`. It is `src/data/preflight.py`, run on the **on-disk** files:

| Check | Pass | Fail |
|---|---|---|
| Eval file | 300 examples, 150 Hotpot + 150 single-hop (NQ / TriviaQA / SQuAD) | Stale / synthetic / prefix-skewed slice |
| Corpus | `len(corpus.jsonl)` ≥ `min_corpus_passages` (50,000) and **no** NQ answer-anchors | Old ~2k index, smoke-test overwrite, or leaked anchors |

On success you see (SQuAD fallback on the committed run; NQ if Tevatron loads):

```text
Ranking data check OK: eval=300 {'hotpot_qa': 150, 'squad': 150} corpus=80000 (>= 50000)
```

On failure the process exits immediately (no GPU download). Re-run `prepare_data.py --hf`. `--limit` still checks the full file and the full corpus — a 50-example debug run on a 2k library is rejected.

`--run-env-check` only rolls a few Gymnasium episodes after the policies run.

Debug cap (keeps the 50/50 mix; do **not** use a prefix `--limit 40`):

```bash
python scripts/run_pilot.py --limit 50 --run-env-check
```

Synthetic / extractive on default.yaml:

```bash
python scripts/run_pilot.py --skip-data-check --limit 50
```

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--limit` | off (full eval) | Stratified cap. On 150+150, `--limit 40` → ~20+20, never 40 Hotpot + 0 NQ. Does **not** skip the 50k corpus check |
| `--split` | `eval` | `eval` or `train` |
| `--policies` | `naive_rag,rule_based,max_tools` | Which policies to run |
| `--reward-preset` | from config (`default`) | Reward weight preset name |
| `--run-env-check` | off | Roll a few Gymnasium episodes with the rule policy (not the data preflight) |
| `--skip-data-check` | off | Skip eval-size + corpus-size preflight (synthetic / extractive debug only) |
| `--no-abstain` | off | Refusal ablation: never emit ABSTAIN (`generation.allow_abstain=false`) |
| `--config` | `configs/default.yaml` | Main config |

**What it does:**
1. Loads corpus + eval examples  
2. **Preflight:** eval mix + corpus ≥ 50k (default.yaml), then stop if stale  
3. Runs **naive RAG** baseline  
4. Runs **agentic** policies  
5. Optionally rolls the RL env on a few examples  
6. Writes trajectories + metric JSONs  

**Console output:** Prints overall, then Hotpot, then SQuAD (or NQ) per policy (not a single mix-weighted JSON blob):
```text
Ranking data check OK: eval=300 {'hotpot_qa': 150, 'squad': 150} corpus=80000 (>= 50000)
Corpus=80000 examples=300 by_dataset={'hotpot_qa': 150, 'squad': 150} preset=default
naive_rag overall: EM=... F1=... reward=... abstain=... n_correct=.../... n_abstained=...
  HotpotQA: ...
  SQuAD: ...
rule_based overall: ...
  HotpotQA: ...
  SQuAD: ...
max_tools overall: ...
  HotpotQA: ...
  SQuAD: ...
env_rollouts: [ {"id": ..., "reward": ..., "em": ..., "f1": ...}, ... ]
Wrote .../results/metrics/pilot_summary_default.json
```

**Artifacts:**

| Path | Contents |
|---|---|
| `results/trajectories/baseline_<preset>.jsonl` | Per-example naive RAG logs |
| `results/trajectories/rule_based_<preset>.jsonl` | Per-example agent trajectories |
| `results/trajectories/max_tools_<preset>.jsonl` | High-cost agent trajectories |
| `results/trajectories/env_rollouts.jsonl` | Gym env check rows (if `--run-env-check`) |
| `results/metrics/baseline_<preset>.json` | Naive summary |
| `results/metrics/rule_based_<preset>.json` | Rule-based summary |
| `results/metrics/max_tools_<preset>.json` | Max-tools summary |
| `results/metrics/pilot_summary_<preset>.json` | Combined table: overall + `results[policy].by_dataset` + `n_examples_by_dataset` |

Each trajectory JSONL row includes: question, prediction, gold, EM/F1, reward components, costs, action history, retrieved passage ids.

---

### 4.4 Reward ablation

**Why:** Show how different reward weights change the scalar `reward` (and later, learned behavior). Required for justifying α/β/γ/λ.

```bash
python scripts/run_reward_ablation.py
```

GPU-tight default is a **stratified 100** from the 300-example eval (50 Hotpot + 50 NQ). Pass `--limit 0` for the full file. This sweep is not the ranking table. Same ranking preflight as the pilot (eval file 300 + corpus ≥ 50k) unless `--skip-data-check`.

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--limit` | 100 | Stratified cap (`0` = full eval). Prefix slicing is gone. Does not skip the corpus preflight |
| `--policy` | `rule_based` | Fixed policy while sweeping rewards |
| `--presets` | all named presets | Comma-separated list from `reward_weights.yaml` |
| `--skip-data-check` | off | Skip eval-size + corpus-size preflight |

**Presets swept by default:**
- `correctness_only`
- `correctness_grounding`
- `correctness_faithfulness_cost`
- `default`
- `lambda_zero`
- `high_cost_pressure`

**Console output:** overall, then Hotpot, then SQuAD (or NQ) for each preset:
```text
default overall: EM=... F1=... reward=... ...
  HotpotQA: ...
  SQuAD: ...
...
Wrote .../results/metrics/reward_ablation_table.json
Wrote .../results/metrics/reward_ablation_by_dataset.json
```

**Artifacts:**

| Path | Contents |
|---|---|
| `results/metrics/ablation_<policy>_<preset>.json` | Full summary per preset |
| `results/metrics/reward_ablation_table.json` | Compact comparison table with nested `by_dataset` per preset |
| `results/metrics/reward_ablation_by_dataset.json` | Sibling table keyed by dataset so plots/docs do not collapse to one EM |

---

## 5. Minimal “first successful run” checklist

1. `smoke_test.py` prints `SMOKE OK` (then re-run `--hf` if you need the ranking corpus; smoke overwrites `data/processed/`)  
2. `prepare_data.py --hf` writes `slice_meta.json` with `source: huggingface_nq_hotpot`, `n_eval: 300`, `eval_by_dataset` 150/150, **`n_passages` ≥ 50000**, **`n_nq_anchor`: 0**. Current ranking snapshot has `nq_corpus: squad` / `nq_hf_dataset: rajpurkar/squad`; a successful Tevatron NQ download would show `dpr_wikipedia_w100` instead.  
3. `run_pilot.py` prints `Ranking data check OK` **before** the model loads (and fails if leftover NQ answer-anchors are present), then writes `pilot_summary_default.json` with three policy blocks, each with `by_dataset`  
4. `run_reward_ablation.py` writes `reward_ablation_table.json` with six presets (**re-run after a corpus swap** — the committed ablation JSON is still the leaked-NQ mix)  
5. `plot_results.py` writes PNGs under `results/figs/`

If anything fails, start from smoke test, then re-prepare data, then re-run pilot.

---

## 6. Related docs

| Doc | Topic |
|---|---|
| `README.md` | Project overview |
| `docs/RESULTS.md` | **Detailed results:** 80k ranking run `e8a4423` (150 Hotpot + 150 SQuAD). Leaked-NQ `2417c43` is historical. |
| `docs/NQ_MAX_TOOLS_ANALYSIS.md` | Leaked-NQ max-tools mechanism; tiny-corpus run `34e6585` (NQ 148/150), not the current SQuAD snapshot |
| `docs/REWARD_DESIGN.md` | Why reward weights were chosen |
| `docs/IMPLEMENTATION_DECISIONS.md` | Verifier = NLI, extractive generator, etc. |
| `docs/EXPERIMENT_LOG.md` | Recorded pilot numbers (each dated block names its run) |
| `docs/data_cards/*.md` | Dataset cards |
