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

Source: `results/metrics/pilot_summary_default.json` — Qwen/Qwen2.5-3B-Instruct, `default` reward, full eval (`limit: null`). Slice: **300 examples (150 Hotpot + 150 NQ)**, corpus 2276 passages (`data/processed/slice_meta.json`). BM25 + lexical NLI.

**Comparison in one line:** on Hotpot, `max_tools` is the worst frozen policy, not a quality ceiling. Extra retrieve/rewrite/rerank/verify cost ~4.7× naive and **lose** exact matches (37 → 34 / 150). Reward ranks **naive > rule > max_tools** because quality is flat-to-worse while spend is not.

Read **Hotpot**, not Overall. NQ is near-ceiling from answer-anchor passages (`The answer is {ans}`). Overall EM is mix-weighted and hides the ranking split.

### HotpotQA (n=150; ranking split)

| Policy | EM | F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | **0.247** | **0.298** | **37/150** | 0.493 | 1.56e-4 | **0.551** |
| rule_based | 0.233 | 0.288 | 35/150 | 0.507 | 4.73e-4 | 0.494 |
| max_tools | 0.227 | 0.285 | 34/150 | 0.493 | 7.12e-4 | 0.435 |

`max_tools` vs `naive_rag` on Hotpot is **4 regressions / 1 recovery** (net −3 EM). All four losses had evidence reordered or swapped after the unconditional rewrite + 3rd retrieve + rerank (on the rewritten query). When the top-5 passages stayed identical to naive (74/150), EM matched naive exactly (0.270). When they changed, max EM dropped. The second retrieve is a no-op (same query → same top-5 on 150/150). Rewrite changed the query on 147/150 Hotpot items and only once recovered a miss (Usher); one rewrite injected a distractor (*Crime and Punishment*) and flipped 1866 → 2011.

`max_tools` is a **high-cost reference**, not a better agent. Unconditional tools pollute context; they do not search more carefully.

### Overall (mix-weighted; do not rank from this)

| Policy | EM | F1 | n_correct | abstain | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.583 | 0.612 | 175/300 | 0.280 | 1.48e-4 | 2.0 | 0.989 |
| rule_based | 0.573 | 0.604 | 172/300 | 0.290 | 4.59e-4 | 4.0 | 0.936 |
| max_tools | 0.567 | 0.600 | 170/300 | 0.283 | 6.95e-4 | 7.0 | 0.872 |

### Natural Questions (n=150; saturated)

| Policy | EM | F1 | n_correct | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.920 | 0.925 | 138/150 | 0.067 | 1.40e-4 | 1.427 |
| rule_based | 0.913 | 0.919 | 137/150 | 0.073 | 4.45e-4 | 1.377 |
| max_tools | 0.907 | 0.915 | 136/150 | 0.073 | 6.77e-4 | 1.310 |

### Cost and action mix

| Policy | retrieve | rewrite | rerank | verify | mean $ | latency | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 1.0 | 0.0 | 0.0 | 0.0 | 1.48e-4 | 496 ms | 640 |
| rule_based | 1.0 | 0.0 | 1.0 | 1.0 | 4.59e-4 | 1032 ms | 1300 |
| max_tools | 3.0 | 1.0 | 1.0 | 1.0 | 6.95e-4 | 1819 ms | 1665 |

### What the three policies show

- **Hotpot quality:** `max_tools` is last (34/150), then rule (35), then naive (37). Extra tools are a quality **regression**, not a ceiling. Three of the four naive→max losses became ABSTAIN after context shuffle.
- **Cost:** spend scales with the frozen scripts (~1× / 3.1× / 4.7× mean $). Latency and tokens follow the same order. The 2nd retrieve in `max_tools` never changes the hit list.
- **Reward:** λ/$ plus slightly worse EM: naive 0.989 > rule 0.936 > max_tools 0.872 (Hotpot 0.551 > 0.494 > 0.435).
- **Grounding vs cost:** `max_tools` has the highest overall `Q_ground` (0.790 vs 0.784 / 0.778) and still fewer gold answers. More passages ≠ better answers when rewrite/rerank reorder the prompt.
- **Abstain:** Hotpot abstain is ~49–51% for all three (parser/policy mix). NQ abstain is ~7%.
- **NQ cannot rank policies.** Extra tools mainly add cost on an answer-anchor ceiling (and still lose 2 EM net vs naive).

Milestone 3 takeaway: a learned controller must **select** tools. Always-on `max_tools` is the wrong operating point on Hotpot.

### Reward-weight ablation (not a ranking table)

Fixed `rule_based` policy, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). EM/F1/$ stay flat (EM 0.56); only the scalar reward changes. Source: `results/metrics/reward_ablation_table.json`.

| Preset | overall reward | Hotpot reward | NQ reward |
|---|---:|---:|---:|
| correctness_only | 0.567 | 0.213 | 0.920 |
| correctness_grounding | 0.868 | 0.427 | 1.308 |
| correctness_faithfulness_cost | 0.862 | 0.386 | 1.338 |
| default | 0.912 | 0.440 | 1.383 |
| lambda_zero | 0.913 | 0.441 | 1.385 |
| high_cost_pressure | 0.889 | 0.383 | 1.396 |

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
