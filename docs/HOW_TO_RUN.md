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
python scripts/smoke_test.py                  # forces extractive (offline)
python scripts/prepare_data.py --hf           # locked 150 Hotpot + 150 NQ (ignores stale 25+25 yaml)
python scripts/run_pilot.py --run-env-check   # refuses to start unless eval file is 300
python scripts/run_reward_ablation.py         # stratified 100 from the same file
python scripts/plot_results.py                # metrics → results/figs/*.png
```

Fast extractive-only debug (not a ranking run):

```bash
python scripts/run_pilot.py --config configs/extractive.yaml --limit 50
```

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
| **Natural Questions** (NQ Open) | `google-research-datasets/nq_open` | Single-hop QA; teaches when *not* to over-retrieve |

### Modes

| Flag | What you get |
|---|---|
| `--hf` | Real NQ + Hotpot slices downloaded from HuggingFace. Hotpot contexts become the shared passage corpus; NQ items get answer-anchor passages in V1 (documented limitation until a full Wikipedia index is added). |
| `--synthetic` | Offline closed fact corpus (capitals/scientists). Use when you have no network or want a perfect EM smoke/pilot. |
| *(no flag)* | Tries HuggingFace first; falls back to synthetic if download fails. |

### Default slice sizes (`configs/default.yaml`)

Locked **plan A** (balanced mix). Ranking reads Hotpot; NQ stays near-ceiling until answer-anchors go away.

| Split | Hotpot | NQ | Typical total |
|---|---|---|---|
| Train | 60 | 40 | 100 |
| Eval | 150 | 150 | **300** |

`--limit` is optional and **stratified** (not a JSONL prefix). Prefix `--limit 40` on this file would be 40 Hotpot + 0 NQ. Ablation defaults to a stratified 100 from the same 300 (labeled as a subset, not the ranking table).

Actual counts after prepare are written to `data/processed/slice_meta.json`.

### Files written by data prep

| File | Contents |
|---|---|
| `data/processed/train_slice.jsonl` | Training questions |
| `data/processed/eval_slice.jsonl` | Evaluation questions |
| `data/processed/corpus.jsonl` | Shared passages for BM25 |
| `data/processed/slice_meta.json` | Counts, source (`huggingface_nq_hotpot` or `synthetic`), paths |

Each example has fields like: `id`, `dataset`, `question`, `answer` / `answers`, `supporting_titles` (Hotpot).

More detail: `docs/data_cards/hotpotqa.md`, `docs/data_cards/natural_questions.md`.

---

## 3. Evaluation matrix (metrics)

The project reports **quality**, **grounding/safety**, **efficiency/cost**, and the **aggregate reward**. This matches Scope Memo V2 §9 (quality–cost trade-off is the headline).

**Always split HotpotQA and Natural Questions.** `aggregate_metrics` groups rows by `dataset` and emits the same means under `by_dataset`. Overall is mix-weighted and is a headline only after you have seen the mix. NQ is currently saturated (answer-anchor passages); do not rank policies from overall EM.

### Quality

| Metric | Meaning |
|---|---|
| **EM** (`mean_em`) | Exact Match after answer normalization (lowercasing, strip articles/punctuation) |
| **Token F1** (`mean_f1`) | Token overlap F1 between prediction and gold |
| **n_correct** | Number of examples with EM = 1 |
| **n_abstained / abstain_rate** | Count and fraction of abstaining examples |
| **by_dataset** | Same fields, grouped by `hotpot_qa` and `natural_questions` |

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
| **rule_based** | Threshold policy over retrieve/rewrite/rerank/verify/stop |
| **max_tools** | Uses tools up to caps (high-cost reference) |

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

**Console output (example for `--hf`):**
```json
{
  "source": "huggingface_nq_hotpot",
  "seed": 42,
  "n_train": 100,
  "n_eval": 300,
  "train_by_dataset": {"hotpot_qa": 60, "natural_questions": 40},
  "eval_by_dataset": {"hotpot_qa": 150, "natural_questions": 150},
  "nq_corpus": "answer_anchor_passages",
  "paths": { ... }
}
Wrote processed slices to .../data/processed
```

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

Debug cap (keeps the 50/50 mix; do **not** use a prefix `--limit 40`):

```bash
python scripts/run_pilot.py --limit 50 --run-env-check
```

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--limit` | off (full eval) | Stratified cap. On 150+150, `--limit 40` → ~20+20, never 40 Hotpot + 0 NQ |
| `--split` | `eval` | `eval` or `train` |
| `--policies` | `naive_rag,rule_based,max_tools` | Which policies to run |
| `--reward-preset` | from config (`default`) | Reward weight preset name |
| `--run-env-check` | off | Roll a few Gymnasium episodes with the rule policy |
| `--no-abstain` | off | Refusal ablation: never emit ABSTAIN (`generation.allow_abstain=false`) |
| `--config` | `configs/default.yaml` | Main config |

**What it does:**
1. Loads corpus + eval examples  
2. Runs **naive RAG** baseline  
3. Runs **agentic** policies  
4. Optionally rolls the RL env on a few examples  
5. Writes trajectories + metric JSONs  

**Console output:** Prints overall, then Hotpot, then NQ per policy (not a single mix-weighted JSON blob):
```text
Corpus=... examples=300 by_dataset={'hotpot_qa': 150, 'natural_questions': 150} preset=default
naive_rag overall: EM=... F1=... reward=... abstain=... n_correct=.../... n_abstained=...
  HotpotQA: ...
  Natural Questions: ...
rule_based overall: ...
  HotpotQA: ...
  Natural Questions: ...
max_tools overall: ...
  HotpotQA: ...
  Natural Questions: ...
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

GPU-tight default is a **stratified 100** from the 300-example eval (50 Hotpot + 50 NQ). Pass `--limit 0` for the full file. This sweep is not the ranking table.

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--limit` | 100 | Stratified cap (`0` = full eval). Prefix slicing is gone |
| `--policy` | `rule_based` | Fixed policy while sweeping rewards |
| `--presets` | all named presets | Comma-separated list from `reward_weights.yaml` |

**Presets swept by default:**
- `correctness_only`
- `correctness_grounding`
- `correctness_faithfulness_cost`
- `default`
- `lambda_zero`
- `high_cost_pressure`

**Console output:** overall, then Hotpot, then NQ for each preset:
```text
default overall: EM=... F1=... reward=... ...
  HotpotQA: ...
  Natural Questions: ...
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

1. `smoke_test.py` prints `SMOKE OK`  
2. `prepare_data.py --hf` writes `slice_meta.json` with `source: huggingface_nq_hotpot`, `n_eval: 300`, `eval_by_dataset` 150/150  
3. `run_pilot.py` writes `pilot_summary_default.json` with three policy blocks, each with `by_dataset`  
4. `run_reward_ablation.py` writes `reward_ablation_table.json` with six presets  

5. `plot_results.py` writes PNGs under `results/figs/`

If anything fails, start from smoke test, then re-prepare data, then re-run pilot.

---

## 6. Related docs

| Doc | Topic |
|---|---|
| `README.md` | Project overview |
| `docs/RESULTS.md` | **Detailed results:** tables, trajectory fields, ablation interpretation |
| `docs/REWARD_DESIGN.md` | Why reward weights were chosen |
| `docs/IMPLEMENTATION_DECISIONS.md` | Verifier = NLI, extractive generator, etc. |
| `docs/EXPERIMENT_LOG.md` | Recorded pilot numbers |
| `docs/data_cards/*.md` | Dataset cards |
