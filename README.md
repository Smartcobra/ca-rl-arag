# CA-RL-ARAG — Milestone 2 Baseline & RL Environment

Cost-Aware Reinforcement Learning for Agentic RAG (V1 implementation).

This package delivers the Milestone 2 checklist from the research roadmap:

- Baseline RAG (`src/rag_baseline.py`)
- Agentic RAG loop with frozen actions `{retrieve, rewrite, rerank, verify, stop}` (`src/agentic_rag.py`)
- Gymnasium RL environment (`src/rag_env.py`)
- Explicit multi-component reward + ablation presets (`src/rewards.py`, `configs/reward_weights.yaml`)
- Dataset slices for **HotpotQA + single-hop** (NQ preferred; TriviaQA / SQuAD fallbacks — Scope Memo V2 §7)
- Pilot logs, metrics, data cards, and implementation decision notes
- Tiny REINFORCE trainer (`src/policies/learned.py`, `scripts/train_policy.py`) — `default` and λ=80 frontier evals are naive RAG; free-cost `correctness_only` used the step cap; λ=0 used tools (EM 0.347, 104/300); λ=20 is EM-tied at 4 steps / 3 retrieve

## Design locks (review comments)

| Decision | Choice | Why |
|---|---|---|
| Verifier | **NLI** (`lexical_nli` default; optional `neural_nli`) | Consistent across experiments; not LLM-as-judge |
| Reward weights | Justified defaults + ablation presets | See `docs/REWARD_DESIGN.md` |
| Complexity order | Frozen baselines first, then tiny REINFORCE | Rule / naive / max ranked; `default` and λ=80 frontier learned evals tied naive. Free-cost sanity used the step cap. λ=0 used tools (EM 0.347); λ=20 EM-tied at 4 steps / 3 retrieve. GRPO/PPO still deferred |
| Action space | Five actions only | V1 discipline; semantic/keyword/expand deferred |

## Quick start

```bash
# from repo root or this folder
cd agentic_rag_rl
source ../.venv/bin/activate   # or your venv
pip install -r requirements.txt   # torch + transformers for Qwen

# 1) offline smoke test (forces extractive; no model download)
python scripts/smoke_test.py

# 2) prepare data (HF builds 300 questions + ~80k-passage distractor pool)
python scripts/prepare_data.py --hf
# offline tiny corpus only: python scripts/prepare_data.py --synthetic

# 3) ranking pilot (Qwen). Preflight runs first: eval must be 300 and corpus >= 50k.
#    If that fails, the script exits before loading the model.
python scripts/run_pilot.py --run-env-check
# extractive debug (no 50k gate): python scripts/run_pilot.py --config configs/extractive.yaml --limit 50
# synthetic / stale files on default.yaml: add --skip-data-check

# 4) reward-weight ablation (same preflight; stratified 100; not the ranking table)
python scripts/run_reward_ablation.py

# 5) REINFORCE on the 100 train examples only — never the 300 eval
python scripts/train_policy.py
# 6) freeze the .pt and score the 300 (the ranking row)
python scripts/run_pilot.py --policies learned

# 7) optional free-cost sanity (separate .pt / curve; see notebooks/FreeCost_Trainer_Sanity_CA_RL_ARAG.ipynb)
python scripts/train_policy.py --reward-preset correctness_only \
  --checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --curve results/metrics/train_policy_curve_correctness_only.json
python scripts/run_pilot.py --reward-preset correctness_only \
  --policies naive_rag,learned \
  --learned-checkpoint results/checkpoints/learned_policy_correctness_only.pt \
  --no-figures
```

**Ranking preflight.** `run_pilot.py` and `run_reward_ablation.py` (default config) refuse to load Qwen unless the **on-disk** eval file is 300 (150 Hotpot + 150 single-hop: NQ, TriviaQA, or SQuAD) **and** `corpus.jsonl` has at least **50,000** passages **and** there are no leftover NQ answer-anchors. `--limit` does not skip this — a 50-question run on a 2k corpus is still the wrong experiment. `--run-env-check` only rolls a few Gym episodes; it is not the data check. Bypass with `--skip-data-check` for synthetic/extractive debug. `configs/extractive.yaml` does not set the 50k floor. The committed ranking snapshot (`d456d26`) used Tevatron/wikipedia-nq.

**Full guide** (why each command, datasets, evaluation matrix, outputs): [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md).  
**Results explained** (pilot tables, train curve, how to read metrics/trajectories, ablations): [`docs/RESULTS.md`](docs/RESULTS.md).

## Pilot results (snapshot)

**Run this section describes:** 80k-passage Qwen ranking, same slice as commit `d456d26` (2026-08-27), **rescored 2026-09-04**. Learned 300-eval: 2026-09-10 (`learned_default.json`; `learned_policy.pt` now local). Free-cost sanity: 2026-09-13 (`learned_correctness_only.json` / `learned_lambda_zero.json`). λ=80 frontier: 2026-09-13 exam, `.pt` synced 2026-09-15/16 (`learned_frontier_act*.json`). Source for frozen baselines: `results/metrics/rule_based_default.json` / `max_tools_default.json` / `baseline_default.json`. Slice: **300 examples (150 Hotpot + 150 NQ)**. Tevatron/wikipedia-nq loaded (`nq_corpus: dpr_wikipedia_w100`, `n_nq_anchor: 0`, 1,450 wiki golds / 847 eval gold articles). Corpus **80,000** passages. BM25 gold recall@5: Hotpot **0.927** (11 misses), NQ **0.587** (62 misses). Lexical NLI. `force_yes_no: true`, `allow_abstain: true`. Current `pilot_summary_default.json` holds naive / rule / max / learned. `--policies learned` merges into that file. The leaked-NQ write-up (148/150) is a different run: [`docs/NQ_MAX_TOOLS_ANALYSIS.md`](docs/NQ_MAX_TOOLS_ANALYSIS.md).

**Comparison in one line:** Hotpot is 59 / 56 / 61 / **59**. NQ is 41 / 41 / 44 / **41**. `learned` (frozen argmax) **is naive RAG**. Extra tools still move a few answers on rule/max, but spend is 4.3× on `max_tools`, so reward ranks **naive = learned > rule > max_tools**.

Read **Hotpot and NQ**, not Overall.

### HotpotQA (n=150; ranking split)

| Policy | EM | F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.631** |
| rule_based | 0.373 | 0.438 | 56/150 | 25 | 0.167 | 4.81e-4 | 0.570 |
| max_tools | **0.407** | **0.485** | **61/150** | 23 | 0.153 | 7.38e-4 | 0.569 |
| learned | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | 0.631 |

Versus the leaked-NQ 80k run: Hotpot stays in the 56–61 band. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). `learned` vs naive is **0 / 0**.

### Natural Questions (n=150; ranking split — not saturated)

| Policy | EM | F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | **0.529** |
| rule_based | 0.273 | 0.352 | 41/150 | 15 | 0.100 | 5.27e-4 | 0.492 |
| max_tools | **0.293** | **0.358** | **44/150** | 18 | 0.120 | 7.55e-4 | 0.450 |
| learned | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | 0.529 |

Rule vs naive is **1 recovery / 1 regression** (tied). Max vs naive is **5 / 2** (net +3). Learned vs naive is **0 / 0** (same 41 answers).

### Overall (mix-weighted; do not rank from this)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | **0.580** |
| rule_based | 0.323 | 0.395 | 97/300 | 40 | 5.04e-4 | 4.0 | 0.531 |
| max_tools | **0.350** | **0.422** | **105/300** | 41 | 7.47e-4 | 7.0 | 0.509 |
| learned | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | 0.580 |

Overall EM dropped **0.68 → 0.33** vs leaked NQ because copy-the-anchor is gone.

### Cost and action mix

| Policy | retrieve | rewrite | rerank | verify | mean $ | latency | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 1.0 | 0.0 | 0.0 | 0.0 | 1.72e-4 | 1046 ms | 795 |
| rule_based | 1.0 | 0.0 | 1.0 | 1.0 | 5.04e-4 | 1765 ms | 1599 |
| max_tools | 3.0 | 1.0 | 1.0 | 1.0 | 7.47e-4 | 3420 ms | 2055 |
| learned | 1.0 | 0.0 | 0.0 | 0.0 | 1.72e-4 | 1169 ms | 795 |

Latency is high because BM25 scores 80k passages per query.

### What the four policies show

- **Retrieval can fail now.** Hotpot recall@5 is 0.927 (11/150). NQ recall@5 is 0.587 (62/150). That was the point of the distractor pool plus real (non-anchor) golds.
- **Hotpot quality is still a few-hit race** (59 / 56 / 61 / 59). Extra retrieves recover 3 and lose 1; not enough to pay 4.3× $. Learned recovers 0.
- **NQ can rank policies on quality, barely.** Max 44 vs naive/rule/learned 41.
- **Abstain** on Hotpot is 29 / 25 / 23 / 29. NQ abstain is 16 / 15 / 18 / 16.
- **Cost still decides reward:** naive = learned 0.580 > rule 0.531 > max_tools 0.509.
- **`learned` is naive on the exam.** Frozen argmax: retrieve→stop 300/300, 0 verify, same 100 answers as naive.

### REINFORCE train curve vs 300-eval

Train (`train_policy_curve.json`, 100 examples) sampled extra tools. Eval (`learned_default.json`, 300) did not.

| Epoch | mean reward | mean EM | mean steps | mean retrieve | mean verify |
|---|---:|---:|---:|---:|---:|
| 1 | **0.649** | 0.42 | 4.75 | 1.62 | 0.46 |
| 2 | 0.620 | 0.39 | 4.20 | 1.39 | 0.46 |
| 3 | 0.577 | 0.36 | 4.45 | 1.41 | 0.55 |
| 4 | 0.564 | 0.36 | 4.30 | 1.44 | 0.50 |
| 5 | 0.643 | 0.41 | 4.01 | 1.30 | 0.39 |

Train verify stayed ~0.4–0.55. Eval verify is **0.0**. Do not compare 0.649 to naive 0.580 — different questions, and train samples while eval uses argmax. The ranking row is the 300-eval: learned = naive.

### Free-cost trainer sanity (2026-09-13; not a ranking table)

Same 100-train / 300-eval. Separate checkpoints. Notebook: `notebooks/FreeCost_Trainer_Sanity_CA_RL_ARAG.ipynb`.

| Preset | Train last steps / retrieve / verify | Eval steps / retrieve / verify | Eval EM (n_correct) |
|---|---|---|---|
| `correctness_only` | 5.84 / 1.84 / 1.20 | **8.0 / 3.0 / 2.0** (300/300 at the step cap) | **0.340 (102/300)** |
| `lambda_zero` | 4.03 / 1.32 / 0.39 | **2.0 / 1.0 / 0.0** (retrieve→stop 300/300) | 0.333 (100/300) |

`correctness_only` vs naive: Hotpot 60 vs 59, NQ 42 vs 41 (9 recoveries / 7 regressions). The trainer can leave two steps when only \(Q_{\mathrm{ans}}\) pays. `lambda_zero` stays naive because \(P_{\mathrm{act}}=0.02\) is still on. Sources: `train_policy_curve_correctness_only.json`, `learned_correctness_only.json`, `train_policy_curve_lambda_zero.json`, `learned_lambda_zero.json`.

### Learned cost-pressure frontier (2026-09-13 failed; λ=0 / 20 scored 2026-09-23)

Same quality terms as `default`. **2026-09-13 / re-synced 2026-09-15:** \(\lambda=80\), `act_penalty` 0 / 0.005 / 0.02. All three learned evals retrieve→stop 300/300. Extra max_tools $ at λ=80 is ~0.046 vs ~0.028 quality, so even `act_penalty=0` was still high pressure. Source: `frontier_sweep_table.json` (still those labels). Checkpoints: `learned_policy_frontier_act*.pt`.

**2026-09-23 (λ=0 / 20, 40 epochs):** `frontier_lambda0` used tools on the 300. `frontier_lambda20` stayed EM-tied to naive but is not retrieve→stop. `frontier_act02` at 40 epochs is not on disk. Notebook: `notebooks/Frontier_Cost_Pressure_CA_RL_ARAG_v2.ipynb`.

| Point | Eval EM | $ | steps / retrieve / verify |
|---|---:|---:|---|
| naive / rule / max (frozen) | 0.333 / 0.323 / **0.350** | 1.72e-4 / 5.04e-4 / 7.47e-4 | 2 / 4 / 7 |
| learned act=0 / 0.005 / 0.02 (λ=80, 5-epoch) | 0.333 | 1.72e-4 | **2.0 / 1.0 / 0.0** |
| learned `frontier_lambda0` | **0.347** (104/300) | 3.41e-4 | **6.0 / 1.0 / 2.0** |
| learned `frontier_lambda20` | 0.333 (100/300) | 2.72e-4 | **4.0 / 3.0 / 0.0** |

λ=0 also rewrote twice on every item (300/300). λ=20 used three retrieves and no rewrite/verify (300/300). Split: λ=0 Hotpot 53 / NQ 51; λ=20 stays 59 / 41. Do not cite `frontier_sweep_table.json` as this family.

### Reward-weight ablation (not a ranking table)

Fixed `rule_based` policy, **stratified 100** from the **same Tevatron-NQ** 300-file (50 Hotpot + 50 NQ). EM/F1/$ stay flat (EM 0.34); only the scalar reward changes. Source: `results/metrics/reward_ablation_table.json` (rescored 2026-09-04). Presets with γ \(Q_{\mathrm{cal}}\) dropped vs the pre-fix table.

| Preset | overall reward | Hotpot reward | NQ reward |
|---|---:|---:|---:|
| correctness_only | 0.362 | 0.373 | 0.351 |
| correctness_grounding | 0.547 | 0.495 | 0.598 |
| correctness_faithfulness_cost | 0.519 | 0.467 | 0.570 |
| default | 0.499 | 0.449 | 0.549 |
| lambda_zero | 0.501 | 0.451 | 0.551 |
| high_cost_pressure | 0.431 | 0.382 | 0.479 |

Full interpretation: [`docs/RESULTS.md`](docs/RESULTS.md). Eval contract: [`docs/IMPLEMENTATION_DECISIONS.md`](docs/IMPLEMENTATION_DECISIONS.md).

<details>
<summary>Legacy extractive, 40-ex, pipeline sanity only (different generator; do not compare to the table above)</summary>

| Policy | EM | F1 | mean $ | retrieves | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.050 | 0.091 | 9.6e-5 | 1.0 | 0.361 |
| rule_based | 0.025 | 0.066 | 3.6e-4 | 1.0 | 0.285 |
| max_tools | 0.075 | 0.143 | 5.4e-4 | 3.0 | 0.303 |

</details>


## Layout

```
agentic_rag_rl/
├── configs/           # default, reward weights, FinOps price card
├── data/processed/    # train/eval slices + corpus
├── docs/              # reward design, decisions, experiment log, data cards
├── notebooks/         # Colab 80k ranking, free-cost sanity, cost-pressure frontier
├── scripts/           # prepare_data, run_pilot, ablation, train_policy, smoke_test
├── src/               # baseline, agent, env, rewards, retrieval, NLI verify
└── results/           # trajectories + metrics
```

## RL environment API

```python
from src.config import load_config
from src.retrieval import BM25Retriever
from src.rag_env import AgenticRAGEnv, ACTION_TO_IDX
from src.utils import read_jsonl

cfg = load_config()
retriever = BM25Retriever(read_jsonl("data/processed/corpus.jsonl"))
examples = read_jsonl("data/processed/eval_slice.jsonl")
env = AgenticRAGEnv(cfg, retriever, examples)
obs, info = env.reset(options={"example": examples[0]})
obs, r, term, trunc, info = env.step(ACTION_TO_IDX["retrieve"])
obs, r, term, trunc, info = env.step(ACTION_TO_IDX["stop"])
```

Sparse episode reward is returned on `stop` with a full component breakdown in `info["episode_result"]`.

## Next (Milestone 3)

- `default` and λ=80 frontier learned evals **tied naive** (retrieve→stop). λ=0 / 20 40-epoch exams are on disk: λ=0 used tools (EM 0.347, 104/300); λ=20 is EM-tied at 4 steps / 3 retrieve. Next: 40-epoch `frontier_act02` and regenerate `frontier_sweep_table.json` / `frontier_em_usd.png`.
- Compare against Adaptive-RAG as the open-loop baseline
- λ–μ Pareto sweeps using `configs/reward_weights.yaml` → `pareto_sweep`
- Optional neural NLI + denser retriever once the extractive/BM25 pipeline is solid
