# CA-RL-ARAG — Milestone 2 Baseline & RL Environment

Cost-Aware Reinforcement Learning for Agentic RAG (V1 implementation).

This package delivers the Milestone 2 checklist from the research roadmap:

- Baseline RAG (`src/rag_baseline.py`)
- Agentic RAG loop with frozen actions `{retrieve, rewrite, rerank, verify, stop}` (`src/agentic_rag.py`)
- Gymnasium RL environment (`src/rag_env.py`)
- Explicit multi-component reward + ablation presets (`src/rewards.py`, `configs/reward_weights.yaml`)
- Dataset slices for **Natural Questions + HotpotQA** (Scope Memo V2 §7)
- Pilot logs, metrics, data cards, and implementation decision notes

## Design locks (review comments)

| Decision | Choice | Why |
|---|---|---|
| Verifier | **NLI** (`lexical_nli` default; optional `neural_nli`) | Consistent across experiments; not LLM-as-judge |
| Reward weights | Justified defaults + ablation presets | See `docs/REWARD_DESIGN.md` |
| Complexity order | Stable baselines **before** GRPO/PPO | Rule-based / naive / max-tools first |
| Action space | Five actions only | V1 discipline; semantic/keyword/expand deferred |

## Quick start

```bash
# from repo root or this folder
cd agentic_rag_rl
source ../.venv/bin/activate   # or your venv
pip install -r requirements.txt   # torch + transformers for Qwen

# 1) offline smoke test (forces extractive; no model download)
python scripts/smoke_test.py

# 2) prepare data (HF if available, else synthetic pilot)
python scripts/prepare_data.py --synthetic
# or: python scripts/prepare_data.py --hf

# 3) run pilot on the full 300-example eval (Qwen/Qwen2.5-3B-Instruct)
python scripts/run_pilot.py --run-env-check
# extractive debug: python scripts/run_pilot.py --config configs/extractive.yaml --limit 50

# 4) reward-weight ablation (stratified 100 from the same eval; not the ranking table)
python scripts/run_reward_ablation.py
```

**Full guide** (why each command, datasets, evaluation matrix, outputs): [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md).  
**Results explained** (pilot tables, how to read metrics/trajectories, ablations): [`docs/RESULTS.md`](docs/RESULTS.md).

## Pilot results (snapshot)

**Not a ranking.** Source: `results/metrics/pilot_summary_default.json` — Qwen/Qwen2.5-3B-Instruct, `default` reward, full eval (`limit: null`). Slice: **300 examples (150 Hotpot + 150 NQ)**, corpus 2276 passages (`data/processed/slice_meta.json`).

Read **Hotpot**, not Overall. NQ is near-ceiling from answer-anchor passages (`The answer is {ans}`). Overall EM is mix-weighted. Hotpot is 43–48 / 150 (gaps of a few items). NQ is 143–144 / 150 and almost flat.

### Overall (mix-weighted headline)

| Policy | EM | F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.633 | 0.684 | 190/300 | 0.160 | 1.43e-4 | 1.059 |
| rule_based | 0.623 | 0.675 | 187/300 | 0.150 | 4.48e-4 | 1.006 |
| max_tools | 0.637 | 0.688 | 191/300 | 0.137 | 6.85e-4 | 0.964 |

### HotpotQA (n=150; ranking split)

| Policy | EM | F1 | n_correct | abstain | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.307 | 0.393 | 46/150 | 0.313 | 0.635 |
| rule_based | 0.287 | 0.374 | 43/150 | 0.300 | 0.568 |
| max_tools | 0.320 | 0.406 | 48/150 | 0.267 | 0.556 |

### Natural Questions (n=150; saturated)

| Policy | EM | F1 | n_correct | abstain | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.960 | 0.974 | 144/150 | 0.007 | 1.482 |
| rule_based | 0.960 | 0.977 | 144/150 | 0.000 | 1.443 |
| max_tools | 0.953 | 0.970 | 143/150 | 0.007 | 1.372 |

NQ cannot rank policies. Extra tools mainly add cost, so overall reward still ranks naive > rule > max_tools.

### Reward-weight ablation (not a ranking table)

Fixed `rule_based` policy, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). EM/F1/$ stay flat; only the scalar reward changes. Source: `results/metrics/reward_ablation_table.json`.

| Preset | overall reward | Hotpot reward | NQ reward |
|---|---:|---:|---:|
| correctness_only | 0.624 | 0.299 | 0.949 |
| correctness_grounding | 0.949 | 0.549 | 1.348 |
| correctness_faithfulness_cost | 0.947 | 0.514 | 1.380 |
| default | 0.977 | 0.535 | 1.418 |
| lambda_zero | 0.978 | 0.536 | 1.420 |
| high_cost_pressure | 0.952 | 0.473 | 1.430 |

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
├── scripts/           # prepare_data, run_pilot, ablation, smoke_test
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

- Compare naive RAG / rule-based / prompted agent / bandit or GRPO policy
- λ–μ Pareto sweeps using `configs/reward_weights.yaml` → `pareto_sweep`
- Optional neural NLI + denser retriever once the extractive/BM25 pipeline is solid
