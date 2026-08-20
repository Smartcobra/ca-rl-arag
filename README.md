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

# 3) run pilot with local Qwen/Qwen2.5-3B-Instruct (default.yaml)
python scripts/run_pilot.py --limit 30 --run-env-check
# extractive-only: python scripts/run_pilot.py --config configs/extractive.yaml --limit 30

# 4) reward-weight ablation table
python scripts/run_reward_ablation.py --limit 30
```

**Full guide** (why each command, datasets, evaluation matrix, outputs): [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md).  
**Results explained** (pilot tables, how to read metrics/trajectories, ablations): [`docs/RESULTS.md`](docs/RESULTS.md).

## Pilot results (snapshot)

**Not a ranking.** The Qwen 40-example headline mixes two regimes: 25 HotpotQA + 15 NQ. NQ EM is **0.933 for every policy** (answer-anchor passages in `prepare_data.py`). Overall EM 0.45 / 0.425 / 0.475 is `(Hotpot correct + 14 NQ correct) / 40`. Every summary, printout, plot, and table now shows Hotpot and NQ separately so NQ padding cannot be treated as a policy result. Hotpot gaps are 1–2 EM hits on 25 examples — still not significant. Ranking waits on n≈250.

Source: `results/metrics/pilot_summary_default.json` (Qwen/Qwen2.5-3B-Instruct, `default` reward, `--limit 40`).

### Overall (mix-weighted headline)

| Policy | EM | F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.450 | 0.488 | 18/40 | 0.325 | 1.45e-4 | 0.788 |
| rule_based | 0.425 | 0.463 | 17/40 | 0.350 | 4.57e-4 | 0.721 |
| max_tools | 0.475 | 0.515 | 19/40 | 0.275 | 6.87e-4 | 0.726 |

### HotpotQA (n=25; only discriminative split)

| Policy | EM | F1 | n_correct | abstain | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.160 | 0.221 | 4/25 | 0.480 | 0.396 |
| rule_based | 0.120 | 0.181 | 3/25 | 0.520 | 0.313 |
| max_tools | 0.200 | 0.264 | 5/25 | 0.400 | 0.358 |

### Natural Questions (n=15; saturated)

| Policy | EM | F1 | n_correct | abstain | mean reward |
|---|---:|---:|---:|---:|---:|
| naive_rag | 0.933 | 0.933 | 14/15 | 0.067 | 1.441 |
| rule_based | 0.933 | 0.933 | 14/15 | 0.067 | 1.399 |
| max_tools | 0.933 | 0.933 | 14/15 | 0.067 | 1.338 |

NQ is identical across policies. That is expected given answer-anchor passages (`The answer is {ans}`). NQ currently cannot rank policies.

Pilot console print and `plot_results.py` (`policy_by_dataset.png`) use the same split. Full interpretation: [`docs/RESULTS.md`](docs/RESULTS.md). Eval contract: [`docs/IMPLEMENTATION_DECISIONS.md`](docs/IMPLEMENTATION_DECISIONS.md).

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
