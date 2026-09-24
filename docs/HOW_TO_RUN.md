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
python scripts/run_pilot.py --run-env-check   # frozen ranking: naive / rule / max_tools on the 300
python scripts/run_reward_ablation.py         # same preflight; stratified 100 from the same file
python scripts/plot_results.py                # metrics → results/figs/*.png
python scripts/train_policy.py                # REINFORCE on the 100 train examples only — never the 300
python scripts/run_pilot.py --policies learned  # freeze the .pt; score the 300 (the ranking row)
```

`--policies learned` is **not** a re-train. `train_policy.py` only writes a homework curve. The exam is this second `run_pilot` command. Needs `results/checkpoints/learned_policy.pt`.

**This checkout already ran the exam.** Source: `results/metrics/learned_default.json` + `results/trajectories/learned_default.jsonl`. Frozen argmax under `default` is **naive RAG**: retrieve→stop on 300/300, verify 0, predictions identical to naive. `pilot_summary_default.json` holds all four policies (naive / rule / max / learned), restored from the frozen `*_default.json` files. `--policies learned` **merges** into that file; it does not drop rule / max.

**Free-cost sanity (2026-09-13) also ran.** Separate checkpoints: `learned_policy_correctness_only.pt` (eval used the step cap: 8 / 3 retrieve / 2 verify) and `learned_policy_lambda_zero.pt` (still retrieve→stop). Notebook: `notebooks/FreeCost_Trainer_Sanity_CA_RL_ARAG.ipynb`. See §4.7.

Fast extractive-only debug (not a ranking run; `extractive.yaml` has no 50k corpus floor):

```bash
python scripts/run_pilot.py --config configs/extractive.yaml --limit 50
```

If you point **default.yaml** at a synthetic or 2k-passage corpus, the ranking scripts exit before loading Qwen. Use `--skip-data-check` only for that debug path.

---

## 1. Why these commands?

| Step | Command | Why run it |
|---|---|---|
| 1 | `smoke_test.py` | Fast offline sanity check. Confirms imports, synthetic data, baseline RAG, agent loop, Gymnasium env, and reward presets work **before** spending time on downloads or pilots. |
| 2 | `prepare_data.py` | Builds the train/eval question slices and BM25 corpus. Without this, pilot/ablation have nothing to evaluate. |
| 3 | `run_pilot.py` | Main Milestone-2 experiment: compare **naive RAG**, **rule-based agent**, and **max-tools agent**; optionally verify the RL env. Produces metrics + trajectory logs. |
| 4 | `run_reward_ablation.py` | Sweeps reward-weight presets on a fixed policy/slice so you can justify α/β/γ/λ choices (reviewer comment) and later write the paper ablation section. |
| 5 | `train_policy.py` | Milestone 3 REINFORCE: tiny MLP on the **100 train examples only**. **40 epochs** by default. Logs sampled mean reward **and** greedy `eval_reward` per epoch. Does **not** open the 300-example eval file. That curve is homework, not the paper table. |
| 6 | `run_pilot.py --policies learned` | **Why this exists:** freeze `learned_policy.pt` and score the **300 eval** questions training never saw. Same Hotpot/NQ table as naive / rule / max_tools. Without this step you cannot claim a learned-policy result. **This run (2026-09-10):** argmax collapsed to retrieve→stop (identical to naive). |
| 7 | `train_policy.py --reward-preset correctness_only` then `lambda_zero` + matching `--policies learned` | **Why this exists:** prove the trainer can leave two steps when cost is free. **This run (2026-09-13):** `correctness_only` eval hit the 8-step cap; `lambda_zero` stayed retrieve→stop. Use `--checkpoint` / `--curve` / `--no-figures` so the `default` ranking artifacts stay put. |
| 8 | `train_policy.py --reward-preset frontier_lambda0` (then `lambda20`, `act02`) + `plot_results.py --frontier` | **Why this exists:** paper headline. One learned policy per (λ, \(P_{\mathrm{act}}\)) that **crosses break-even**. **On disk (2026-09-23/24, 40 epochs):** λ=0 used tools (EM 0.347, 6 steps); λ=20 is EM-tied at 4 steps / 3 retrieve; λ=80 `act02` is retrieve→stop. Table + `frontier_em_usd.png` regenerated 2026-09-24. Notebook: `Frontier_Cost_Pressure_CA_RL_ARAG_v2.ipynb`. |

They are sequenced so you never debug data/reward issues on a broken pipeline. Train after the frozen ranking exists. Score the learned policy **after** the `.pt` exists; do not mix the train curve into the ranking table.

---

## 2. Datasets used

Primary datasets follow **Scope Memo V2 §7**:

| Dataset | HuggingFace id (when `--hf`) | Role |
|---|---|---|
| **HotpotQA** (distractor) | `hotpotqa/hotpot_qa` | Multi-hop QA; supporting titles enable grounding metrics |
| **Natural Questions** (preferred) | `Tevatron/wikipedia-nq` (DPR Wikipedia 100-word passages) | Single-hop QA with real evidence; teaches when *not* to over-retrieve. |
| **SQuAD / TriviaQA** (fallback) | `rajpurkar/squad` or Tevatron TriviaQA/SQuAD | Same single-hop slot if Tevatron NQ is too heavy. Historical ranking run `e8a4423` used SQuAD. Current ranking slice `d456d26` used NQ (reward rescored 2026-09-04). |

### Modes

| Flag | What you get |
|---|---|
| `--hf` | Real Hotpot + single-hop (DPR Wikipedia NQ if it loads; else TriviaQA / SQuAD), plus an unused-Hotpot **distractor pool** (target 80k, floor 50k). Locked eval stays 150+150. Golds are real passages, not `{question} The answer is {gold}`. Check `slice_meta.json` (`nq_corpus`, `n_nq_anchor`). |
| `--synthetic` | Offline closed fact corpus (capitals/scientists). Use when you have no network or want a perfect EM smoke/pilot. |
| *(no flag)* | Tries HuggingFace first; falls back to synthetic if download fails. |

### Default slice sizes (`configs/default.yaml`)

Locked **plan A** (balanced mix). After rebuilding with `--hf`, single-hop uses real passages (NQ / TriviaQA / SQuAD) and can rank stop vs over-retrieve. Historical RESULTS tables from the answer-anchor NQ corpus (`2417c43`) and the SQuAD fallback (`e8a4423`) are different runs. Current on-disk metrics are **150 Hotpot + 150 NQ** (slice `d456d26`, reward/\(Q_{\mathrm{cal}}\) rescored 2026-09-04).

| Split | Hotpot | Single-hop (NQ or fallback) | Typical total |
|---|---|---|---|
| Train | 60 | 40 | 100 |
| Eval | 150 | 150 | **300** |

`--limit` is optional and **stratified** (not a JSONL prefix). Prefix `--limit 40` on this file would be 40 Hotpot + 0 NQ. Ablation defaults to a stratified 100 from the same 300 (labeled as a subset, not the ranking table).

Actual counts after prepare are written to `data/processed/slice_meta.json`. After `--hf` you should see `n_eval: 300`, `n_passages` ≥ 50,000 (usually 80,000), and **`n_nq_anchor`: 0**. The committed ranking snapshot has `nq_corpus: dpr_wikipedia_w100` and `nq_hf_dataset: Tevatron/wikipedia-nq`. Open that file with your own eyes before a Qwen run. A ~2,276-passage file is the old tiny index — do not start a Qwen ranking run on it. A single-hop eval with ~7 distinct gold articles is the old prefix-slice bug — do not start a Qwen ranking run on that either.

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
| **Q_cal** | Calibration: justified abstain vs lazy refuse / confident wrong. Numeric table: [`REWARD_DESIGN.md`](REWARD_DESIGN.md) |
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
| **learned** | Frozen REINFORCE checkpoint. Same eval loop as the others. **Scoring only.** On the 2026-09-10 300-eval, frozen argmax was retrieve→stop 300/300 (identical to naive). Skipped if the checkpoint is missing. |

**Verifier vs frozen policy (read this before RL).** Lexical NLI is not a dummy label. On the current Tevatron-NQ ranking run (`d456d26`), Hotpot was 14 contradiction / 31 neutral / 105 support and NQ was 0 / 18 / 132. After every `verify` — including all 14 Hotpot contradictions and 18 NQ neutrals — `rule_based` just stops. Details: [`RESULTS.md` §9](RESULTS.md) and [`IMPLEMENTATION_DECISIONS.md`](IMPLEMENTATION_DECISIONS.md). The leaked-NQ 150/150-support table (`2417c43`) is historical.

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

**Console output (example for `--hf`; committed ranking run used Tevatron NQ):**
```json
{
  "source": "huggingface_nq_hotpot",
  "seed": 42,
  "n_train": 100,
  "n_eval": 300,
  "n_passages": 80000,
  "n_hotpot_distractor": 74955,
  "n_nq_wiki": 1450,
  "n_nq_anchor": 0,
  "train_by_dataset": {"hotpot_qa": 60, "natural_questions": 40},
  "eval_by_dataset": {"hotpot_qa": 150, "natural_questions": 150},
  "nq_corpus": "dpr_wikipedia_w100",
  "nq_hf_dataset": "Tevatron/wikipedia-nq",
  "retrieval_diag": {
    "by_dataset": {
      "hotpot_qa": {"recall@5": 0.927},
      "natural_questions": {"recall@5": 0.587}
    }
  },
  "paths": { ... }
}
Wrote processed slices to .../data/processed
```

Always read `slice_meta.json` rather than assuming NQ. If Tevatron fails, `nq_corpus` is `trivia_qa` or `squad`. Do not start a GPU job on a 7-article or 16-article single-hop prefix.

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

On success you see (Tevatron NQ on the committed run):

```text
Ranking data check OK: eval=300 {'hotpot_qa': 150, 'natural_questions': 150} corpus=80000 (>= 50000)
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
| `--policies` | `naive_rag,rule_based,max_tools,learned` | Which policies to run. `learned` is skipped (with a message) if the checkpoint is missing |
| `--learned-checkpoint` | from `policy.learned.checkpoint` | Eval-only path to the trained MLP. Does **not** train |
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
learned overall: ...
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
| `results/trajectories/learned_<preset>.jsonl` | Frozen REINFORCE checkpoint on the **eval** split (if the `.pt` exists) |
| `results/trajectories/env_rollouts.jsonl` | Gym env check rows (if `--run-env-check`) |
| `results/metrics/baseline_<preset>.json` | Naive summary |
| `results/metrics/rule_based_<preset>.json` | Rule-based summary |
| `results/metrics/max_tools_<preset>.json` | Max-tools summary |
| `results/metrics/learned_<preset>.json` | Learned summary (`by_dataset` same schema) |
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

**Console output:** overall, then Hotpot, then NQ (or SQuAD) for each preset:
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

### 4.5 Train policy (REINFORCE, train split only)

**Why:** Learn a closed-loop controller over `{retrieve, rewrite, rerank, verify, stop}` from the existing 10-d env observation. This is vanilla REINFORCE with a **tiny** MLP (10 → 16 → 5, 261 parameters), not GRPO/PPO. The win condition is using verify `support` / `contradiction` when it is worth the cost — something `rule_based` never does.

**Hard rule:** this script reads `data/processed/train_slice.jsonl` (60 Hotpot + 40 NQ = 100) and **never** opens `eval_slice.jsonl`. There is no `--split` flag. A path whose filename contains `eval` is rejected.

**Scoring the 300 (separate job):** after a checkpoint exists, `run_pilot.py` loads it as a **fourth frozen policy** on `--split eval` (default). That is inference, not training. Same `format_eval_summary` / `by_dataset` table as naive / rule / max_tools.

```bash
python scripts/train_policy.py
```

Extractive laptop / smoke path (does not write a ranking number):

```bash
python scripts/train_policy.py --config configs/extractive.yaml --limit 8 --epochs 2 --skip-data-check
```

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 40 (from `policy.learned.epochs`) | Passes over the 100 train examples. Use 30–50 for ranking / frontier; extractive debug still passes `--epochs 2`. |
| `--lr` | 0.003 | Adam step size |
| `--hidden` | 16 | MLP width. **Refused above 32** |
| `--entropy-coef` | 0.01 | Keeps the policy from collapsing to retrieve→stop on epoch 1 |
| `--limit` | off (full train file) | Stratified cap on **train** only |
| `--skip-data-check` | off | Skip train-size + corpus-size preflight (synthetic / extractive debug only) |
| `--checkpoint` | `results/checkpoints/learned_policy.pt` | Last-epoch weights; best **greedy** `eval_reward` copy is `<stem>_best.pt`. Resume also writes `<stem>_trainer.pt` (epoch, optimizer, baseline, `best_eval_reward`). |
| `--curve` | derived from the checkpoint stem | Learning-curve JSON. Default checkpoint keeps `train_policy_curve.json`; a named `.pt` writes `train_policy_curve_<suffix>.json` |
| `--resume` | off | Load the last `.pt`, `_trainer.pt`, and existing curve, then continue from the next epoch. Use after a Colab disconnect. |
| `--reward-preset` | from config (`default`) | `default` is the ranking train. `correctness_only` / `lambda_zero` are the 2026-09-13 sanity. Frontier family is `frontier_lambda0` / `frontier_lambda20` / `frontier_act02`. Calibration rule is still the 2026-09-04 fix. |

**What it does:**
1. Loads corpus + **train** examples (refuses eval paths)
2. Preflight: train file is 100 (60+40) and corpus ≥ 50k / no NQ anchors (default.yaml)
3. Shares one generator with `AgenticRAGEnv`, same as the pilot
4. Each example: `reset(options={"example": ...})` → sample actions → sparse terminal reward → REINFORCE update (EMA baseline)
5. After the sampled pass, **greedy (argmax) rollout** of the same 100 train examples. Logs `eval_reward` / `eval_em` / `eval_n_steps` / `eval_n_retrieve` / `eval_n_verify` with `"eval_split": "train_greedy"`. Does **not** open `eval_slice.jsonl`.
6. Prints sampled **and** greedy columns per epoch
7. Writes the learning-curve JSON and checkpoints (`_best.pt` = best greedy `eval_reward`)

**Console output (this checkout's first run; `results/metrics/train_policy_curve.json` — 5 epochs, no greedy column yet):**
```text
Ranking data check OK: train=100 {'hotpot_qa': 60, 'natural_questions': 40} corpus=80000 (>= 50000)
TRAIN ONLY path=.../train_slice.jsonl n=100 by_dataset={...} preset=default hidden=16 epochs=5 lr=0.003
epoch 1/5  n=100  mean_reward=0.6489  mean_em=0.420  mean_steps=4.75  mean_retrieve=1.62  mean_verify=0.46
epoch 2/5  n=100  mean_reward=0.6195  mean_em=0.390  mean_steps=4.20  mean_retrieve=1.39  mean_verify=0.46
epoch 3/5  n=100  mean_reward=0.5771  mean_em=0.360  mean_steps=4.45  mean_retrieve=1.41  mean_verify=0.55
epoch 4/5  n=100  mean_reward=0.5641  mean_em=0.360  mean_steps=4.30  mean_retrieve=1.44  mean_verify=0.50
epoch 5/5  n=100  mean_reward=0.6429  mean_em=0.410  mean_steps=4.01  mean_retrieve=1.30  mean_verify=0.39
Wrote .../results/metrics/train_policy_curve.json
Wrote .../results/checkpoints/learned_policy.pt
```

Best train reward was epoch 1 (0.649). Last epoch is 0.643. The curve did not collapse to retrieve→stop (verify stayed 0.39–0.55). This is **train-only**. New trains (2026-09-15+) are **40 epochs** and print `eval_reward` / `eval_steps` / `eval_verify` (argmax on the same 100) next to the sampled columns. The 300-eval `learned` row **is** in `pilot_summary_default.json` (four policies). This checkout now has `learned_policy.pt` plus the free-cost and λ=80 frontier `.pt` files.

**Artifacts:**

| Path | Contents |
|---|---|
| `results/metrics/train_policy_curve.json` | One object per epoch: sampled `mean_reward` **and** greedy `eval_reward` (`"eval_split": "train_greedy"`), plus EM / action counts, `"split": "train"` |
| `results/checkpoints/learned_policy.pt` | Last-epoch MLP weights |
| `results/checkpoints/learned_policy_best.pt` | Best **greedy** `eval_reward` so far |

This curve is a **train** learning signal, not a ranking table. Scoring is the next command (§4.6). Do not claim a policy win from `train_policy_curve.json`. The 300-eval is on disk (`learned_default.json`): frozen argmax is retrieve→stop, identical to naive.

Free-cost trains (2026-09-13) wrote `train_policy_curve_correctness_only.json` (steps 4.66 → 5.84, verify 0.50 → 1.20) and `train_policy_curve_lambda_zero.json` (looks like `default`). See §4.7.

---

### 4.6 Score the learned policy (eval only; not training)

**Why:** `train_policy.py` never saw the 300. The train curve can look good because the MLP sampled extra tools under entropy. `--policies learned` is the exam: load the checkpoint **frozen** (argmax, no gradient) and run the same eval loop as naive / rule / max_tools. That is the only number that can sit next to naive 0.580 / rule 0.531 / max_tools 0.509.

Needs `results/checkpoints/learned_policy.pt`.

```bash
python scripts/run_pilot.py --policies learned
```

`--policies learned` means **only score the new row**. It **merges** into the existing `pilot_summary_default.json`. It does not drop rule / max.

The canonical table is four policies: naive, rule, max, learned. After a learned-only exam the other rows stay. After a reward or slice change, rerun the default four-policy pilot so every row shares the new setup (needs `learned_policy.pt`):

```bash
python scripts/run_pilot.py --run-env-check
```

This checkout **has** the `.pt` files (`learned_policy.pt`, free-cost, and λ=80 frontier). The four-row summary is `pilot_summary_default.json` (naive / rule / max / learned; same ranking numbers).

**This run (2026-09-10).** Source: `results/metrics/learned_default.json`, `results/trajectories/learned_default.jsonl` (300 rows).

```text
learned overall: EM=0.333 F1=0.397 reward=0.580 abstain=0.150 n_correct=100/300 n_abstained=45
  HotpotQA: EM=0.393 n_correct=59/150 retrieve=1.0 verify=0.0 steps=2.0
  Natural Questions: EM=0.273 n_correct=41/150 retrieve=1.0 verify=0.0 steps=2.0
```

| Check | Result |
|---|---|
| Action mix | **retrieve → stop on 300/300**. verify=0, rewrite=0, rerank=0 |
| vs naive | EM/F1/predictions **identical** (300/300 same answers). Reward 0.580 vs 0.580 (latency noise only) |
| vs train curve | Train sampled verify 0.39–0.55 and ~4 steps. Eval **argmax** does not. Entropy hid a naive mode. |

That is a real ranking result, not a missing file. The intern practiced with extra tools; on the exam it copies naive RAG. The Milestone-3 win condition (act on verify `contradiction` / `neutral`) did **not** happen — `verify_out` is null on every learned row.

If the `.pt` is missing:

```text
learned checkpoint missing: .../learned_policy.pt. Train first: python scripts/train_policy.py
```

With `--policies learned` only, the script **exits** if the checkpoint is missing. With the default four-policy list, it **skips** `learned` and still writes naive / rule / max_tools (and **keeps** a previous `learned` row if the summary already had one).

**What to look at:** overall, then Hotpot, then NQ — same as §4.3. You are not trying to beat Qwen. You are trying to beat the **controllers**. On this snapshot, learned **did not**: it matched naive on quality and cost, and never called verify.

**Artifacts:**

| Path | Contents |
|---|---|
| `results/metrics/learned_default.json` | Learned-only summary (`by_dataset` same schema) |
| `results/metrics/pilot_summary_default.json` | Combined four-policy table. This run’s keys overwrite matching rows; other policies are kept |
| `results/trajectories/learned_default.jsonl` | Per-question logs on the **eval** split |

`--learned-checkpoint` points eval at a non-default `.pt`. `--no-figures` skips rewriting `results/figs/` (use this on the free-cost presets so the ranking plots stay the `default` table).

---

### 4.7 Free-cost trainer sanity (`correctness_only` vs `lambda_zero`)

**Why:** under `default`, extra tools are −EV, so retrieve→stop is the correct greedy policy. That does not tell you whether REINFORCE can learn tools at all. Train once with cost off, freeze, score the 300.

Two steps = `retrieve → stop` (naive RAG). Notebook: `notebooks/FreeCost_Trainer_Sanity_CA_RL_ARAG.ipynb`.

```bash
python scripts/train_policy.py --reward-preset correctness_only \
  --checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --curve results/metrics/train_policy_curve_correctness_only.json
python scripts/run_pilot.py --reward-preset correctness_only \
  --policies naive_rag,learned \
  --learned-checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --no-figures
# repeat with --reward-preset lambda_zero and *_lambda_zero paths
```

**This run (2026-09-13).**

| Preset | Train last epoch | Frozen 300-eval | Verdict |
|---|---|---|---|
| `correctness_only` | steps 5.84, retrieve 1.84, verify 1.20 | **8.0 / 3.0 / 2.0 verify**, EM 0.340 (102/300) | Trainer works; used the step cap |
| `lambda_zero` | steps 4.03, verify 0.39 (like `default`) | **2.0 / 1.0 / 0 verify**, identical to naive | \(P_{\mathrm{act}}\) still collapses the policy |

`correctness_only` vs naive: 9 recoveries / 7 regressions (Hotpot 2/1, NQ 7/6). Dominant eval path (245/300): retrieve → rewrite ×2 → verify → retrieve ×2 → verify → stop.

**Artifacts:**

| Path | Contents |
|---|---|
| `results/metrics/train_policy_curve_correctness_only.json` | Train epochs for the free-tools preset |
| `results/metrics/train_policy_curve_lambda_zero.json` | Train epochs with λ=μ=0, \(P_{\mathrm{act}}\) on |
| `results/checkpoints/learned_policy_correctness_only.pt` | Last-epoch weights (plus `_best.pt`) |
| `results/checkpoints/learned_policy_lambda_zero.pt` | Last-epoch weights (plus `_best.pt`) |
| `results/metrics/learned_correctness_only.json` | 300-eval summary |
| `results/metrics/learned_lambda_zero.json` | 300-eval summary |
| `results/trajectories/learned_correctness_only.jsonl` | 300 rows, all 8 steps |
| `results/trajectories/learned_lambda_zero.jsonl` | 300 rows, all retrieve→stop |

These are **not** ranking rows. The paper table still uses `learned_default.json`.

---

### 4.8 Learned cost-pressure frontier (paper headline)

**Why:** one learned policy that beats naive is a weak claim. Train the **same** tiny MLP at three (λ, \(P_{\mathrm{act}}\)) points that **straddle the extra-spend break-even line**. Score each frozen checkpoint on the locked 300. Plot **EM vs mean $** next to naive / rule / max_tools. **Previous run (2026-09-13):** all three learned evals stacked on naive — every knob was λ=80, so extra max_tools $ (~0.046) beat the ~0.028 quality gain even at `act_penalty=0`.

Quality weights stay at `default`. New family:

| Preset | λ | \(P_{\mathrm{act}}\) | Extra $ term vs ~0.028 quality |
|---|---:|---:|---|
| `frontier_lambda0` | 0 | 0 | 0 — tools free |
| `frontier_lambda20` | 20 | 0 | ≈ 0.012 — still +EV |
| `frontier_act02` | 80 | 0.02 | ≈ 0.046 — −EV (expensive end) |

**40 epochs** per preset. Each epoch logs greedy `eval_reward` on the train split. Notebook: `notebooks/Frontier_Cost_Pressure_CA_RL_ARAG.ipynb`.

```bash
python scripts/train_policy.py --reward-preset frontier_lambda0 --epochs 40 \
  --checkpoint results/checkpoints/learned_policy_frontier_lambda0.pt \
  --curve results/metrics/train_policy_curve_frontier_lambda0.json
python scripts/run_pilot.py --reward-preset frontier_lambda0 \
  --policies learned \
  --learned-checkpoint results/checkpoints/learned_policy_frontier_lambda0.pt \
  --no-figures
# repeat for frontier_lambda20 and frontier_act02
# after a disconnect, continue the same checkpoint/curve:
python scripts/train_policy.py --reward-preset frontier_lambda0 --epochs 40 --resume \
  --checkpoint results/checkpoints/learned_policy_frontier_lambda0.pt \
  --curve results/metrics/train_policy_curve_frontier_lambda0.json
python scripts/plot_results.py --frontier
```

`--policies learned` does **not** re-run naive. Frozen dots come from `baseline_default.json` / `rule_based_default.json` / `max_tools_default.json`.

**This run (2026-09-13, old family; `.pt` local).** `frontier_act0` / `act005` / `act02` (all λ=80) frozen evals are retrieve→stop **300/300**, EM 0.333, predictions identical to naive. Those artifacts stay on disk as the failed sweep. **Do not re-train them.**

**This run (2026-09-23/24, λ=0 / 20 / 80, 40 epochs).** Frozen argmax on the locked 300:

| Preset | EM | n_correct | $ | steps / retrieve / rewrite / verify | vs naive |
|---|---:|---:|---:|---|---|
| `frontier_lambda0` | **0.347** | **104/300** | 3.41e-4 | **6.0 / 1.0 / 2.0 / 2.0** (300/300) | +4 exact; Hotpot 53, NQ 51 |
| `frontier_lambda20` | 0.333 | 100/300 | 2.72e-4 | **4.0 / 3.0 / 0.0 / 0.0** (300/300) | EM-tied (59+41); not retrieve→stop |
| `frontier_act02` | 0.333 | 100/300 | 1.72e-4 | **2.0 / 1.0 / 0.0 / 0.0** (300/300) | retrieve→stop; Hotpot 59, NQ 41 |

Last-epoch train: λ=0 sampled 4.98 steps / 1.04 verify / reward 0.720, greedy 6 / 2 verify; λ=20 sampled 4.01 / 0.52 / 0.711, greedy 4 / 3 retrieve / 0 verify; `act02` sampled 2.09 / 0 verify / 0.695, greedy 2 / 1 retrieve on all 40 epochs.

**Artifacts on disk (old λ=80 `act0` / `act005`, 5 epochs):** `learned_frontier_act0.json` / `_act005.json` and matching curves / `.pt`. Do not re-train those two. The 5-epoch `act02` curve is in `results/metrics/partial_curve_backup/`.

**Artifacts on disk (new family, λ=0 / 20 / 80):**

| Path | Contents |
|---|---|
| `results/checkpoints/learned_policy_frontier_lambda0.pt` / `_lambda20.pt` / `_act02.pt` (+ `_best.pt`) | 40-epoch MLP per pressure |
| `results/metrics/learned_frontier_lambda0.json` / `_lambda20.json` / `_act02.json` | 300-eval per pressure |
| `results/metrics/train_policy_curve_frontier_lambda0.json` / `_lambda20.json` / `_act02.json` | 40-epoch sampled + greedy columns |
| `results/metrics/pilot_summary_frontier_*.json` | Learned-only summaries under each preset |
| `results/metrics/frontier_sweep_table.json` | Combined EM / $ / steps (regenerated 2026-09-24) |
| `results/figs/frontier_em_usd.png` | Headline scatter (regenerated 2026-09-24) |
| `results/trajectories/learned_frontier_lambda0.jsonl` / `_lambda20.jsonl` / `_act02.jsonl` | 300 rows each |

Does not overwrite `learned_policy.pt`. Ranking plots were refreshed from `pilot_summary_default.json` at the same time.

---

## 5. Minimal “first successful run” checklist

1. `smoke_test.py` prints `SMOKE OK` (then re-run `--hf` if you need the ranking corpus; smoke overwrites `data/processed/`)  
2. `prepare_data.py --hf` writes `slice_meta.json` with `source: huggingface_nq_hotpot`, `n_eval: 300`, `eval_by_dataset` 150/150, **`n_passages` ≥ 50000**, **`n_nq_anchor`: 0**. Current ranking snapshot has `nq_corpus: dpr_wikipedia_w100` / `nq_hf_dataset: Tevatron/wikipedia-nq` and NQ recall@5 **0.587** (below 1.0).  
3. `run_pilot.py` prints `Ranking data check OK` **before** the model loads, then writes per-policy JSON. `--policies learned` (2026-09-10) wrote `learned_default.json`: retrieve→stop 300/300, identical to naive. That command now **merges** into `pilot_summary_default.json` (it does not drop rule / max). A four-policy rerun still rebuilds every row if the reward or slice changed.  
4. `run_reward_ablation.py` writes `reward_ablation_table.json` with six presets (**re-run after a corpus swap or a reward-formula change** — the on-disk ablation JSON is the Tevatron-NQ stratified 100, rescored 2026-09-04)  
5. `plot_results.py` writes PNGs under `results/figs/`
6. `train_policy.py` printed `TRAIN ONLY` on the 100-example train file and wrote `train_policy_curve.json` (historical: 5 epochs, reward 0.649 → 0.564 → 0.643; **new trains are 40 epochs** with a greedy `eval_reward` column). It never reads `eval_slice.jsonl`.
7. `run_pilot.py --policies learned` wrote `learned_default.json` + `learned_default.jsonl`. **Ranking row:** EM 0.333 / 59+41 correct, retrieve→stop 300/300, identical to naive. Do not treat the train curve (step 6) as this row.
8. Free-cost sanity (2026-09-13) wrote `learned_correctness_only.json` (8 steps / 102 correct) and `learned_lambda_zero.json` (2 steps / 100 correct). Trainer is not stuck; \(P_{\mathrm{act}}\) is.
9. Frontier family (2026-09-23/24, 40 epochs) wrote `learned_frontier_lambda0.json` (EM 0.347 / 6 steps), `_lambda20.json` (EM 0.333 / 4 steps / 3 retrieve), and `_act02.json` (retrieve→stop, EM 0.333). Combined table + figure: `frontier_sweep_table.json` / `frontier_em_usd.png` (regenerated 2026-09-24). Historical 5-epoch `act0` / `act005` stay on disk. Do not overwrite `learned_policy.pt`.

If anything fails, start from smoke test, then re-prepare data, then re-run pilot. Do not debug the trainer against the 300-eval until `correctness_only` has also collapsed.

---

## 6. Related docs

| Doc | Topic |
|---|---|
| `README.md` | Project overview |
| `docs/RESULTS.md` | **Detailed results:** 80k ranking slice `d456d26` (150 Hotpot + 150 NQ), reward/\(Q_{\mathrm{cal}}\) rescored 2026-09-04. §6 `default` train + 300-eval (collapsed to naive), 2026-09-13 free-cost sanity, failed λ=80 `act_penalty` sweep, and 2026-09-23/24 λ=0 / 20 / 80 40-epoch family (tools / 3-retrieve / retrieve→stop). SQuAD `e8a4423` and leaked-NQ `2417c43` are historical. |
| `docs/NQ_MAX_TOOLS_ANALYSIS.md` | Leaked-NQ max-tools mechanism; tiny-corpus run `34e6585` (NQ 148/150), not the current Tevatron-NQ snapshot |
| `docs/REWARD_DESIGN.md` | Why reward weights were chosen |
| `docs/IMPLEMENTATION_DECISIONS.md` | Verifier = NLI, extractive generator, tiny REINFORCE trainer (2026-09-05); train 2026-09-09; learned eval 2026-09-10; free-cost sanity and λ=80 frontier 2026-09-13; frontier retarget 2026-09-15; λ=0 / 20 / 80 40-epoch family 2026-09-23/24 |
| `docs/WHY_SMALL_MLP.md` | Why the policy is 261 parameters; `default` and λ=80 frontier evals were naive; `correctness_only` used the step cap |
| `docs/EXPERIMENT_LOG.md` | Recorded pilot numbers (each dated block names its run) |
| `docs/data_cards/*.md` | Dataset cards |
