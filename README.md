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

Source: `results/metrics/pilot_summary_default.json` — Qwen/Qwen2.5-3B-Instruct, `default` reward, full eval (`limit: null`). Slice: **300 examples (150 Hotpot + 150 NQ)**, corpus 2276 passages. BM25 + lexical NLI. This run includes **forced yes/no** and a **tighter ABSTAIN prompt** (`force_yes_no: true`, `allow_abstain: true`).

**Comparison in one line:** Hotpot is still a 1–2 hit race (60 / 58 / 59). Extra tools still do not beat naive. Reward ranks **naive > rule > max_tools** (~1× / 3.1× / 4.6×). NQ is 148/150 for every policy after the yes/no detector fix.

Read **Hotpot**, not Overall. NQ is saturated from answer-anchor passages.

### HotpotQA (n=150; ranking split)

| Policy | EM | F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | **0.400** | 0.488 | **60/150** | 20 | 0.133 | 1.60e-4 | **0.691** |
| rule_based | 0.387 | 0.481 | 58/150 | 17 | 0.113 | 4.81e-4 | 0.628 |
| max_tools | 0.393 | **0.494** | 59/150 | 19 | 0.127 | 7.20e-4 | 0.593 |

`max_tools` vs naive is **2 regressions / 1 recovery** (net −1). Hotpot is unchanged vs the last Qwen run.

### Overall (mix-weighted; do not rank from this)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.693 | 0.743 | 208/300 | 20 | 1.52e-4 | 2.0 | 1.102 |
| rule_based | 0.687 | 0.740 | 206/300 | 17 | 4.67e-4 | 4.0 | 1.050 |
| max_tools | 0.690 | 0.746 | 207/300 | 19 | 7.03e-4 | 7.0 | 1.002 |

### Natural Questions (n=150; saturated)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.987 | 0.998 | 148/150 | 0 | 1.45e-4 | 1.514 |
| rule_based | 0.987 | 0.998 | 148/150 | 0 | 4.54e-4 | 1.472 |
| max_tools | 0.987 | 0.998 | 148/150 | 0 | 6.86e-4 | 1.411 |

NQ **147 → 148** on all three: `is there a name for the at symbol` now answers `commercial at` (yes/no detector no longer forces `no`). Remaining misses are `in the Gospel of Luke` / `in Santa Monica` (span vs `in …`). Max EM worse than naive on NQ: **0**.

### Cost and action mix

| Policy | retrieve | rewrite | rerank | verify | mean $ | latency | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 1.0 | 0.0 | 0.0 | 0.0 | 1.52e-4 | 502 ms | 666 |
| rule_based | 1.0 | 0.0 | 1.0 | 1.0 | 4.67e-4 | 1083 ms | 1352 |
| max_tools | 3.0 | 1.0 | 1.0 | 1.0 | 7.03e-4 | 1856 ms | 1717 |

### What the three policies show

- **Hotpot quality is a 1–2 hit race** (60 / 58 / 59). Unchanged by the yes/no detector tightening.
- **Abstain** on Hotpot is 20 / 17 / 19 (~12–13%). NQ abstain is 0.
- **Cost still decides reward:** naive 1.102 > rule 1.050 > max_tools 1.002 (Hotpot 0.691 > 0.628 > 0.593).
- **NQ cannot rank policies.** All three are 148/150. The +1 is the at-symbol name question, not tools.

Milestone 3 takeaway: a learned controller must **select** tools. Quality is no longer hidden under 50% abstain; extra tools still do not pay for themselves.

### Reward-weight ablation (not a ranking table)

Fixed `rule_based` policy, **stratified 100** from the same 300-file (50 Hotpot + 50 NQ). EM/F1/$ stay flat (EM 0.69); only the scalar reward changes. Source: `results/metrics/reward_ablation_table.json`.

| Preset | overall reward | Hotpot reward | NQ reward |
|---|---:|---:|---:|
| correctness_only | 0.715 | 0.442 | 0.989 |
| correctness_grounding | 0.991 | 0.593 | 1.389 |
| correctness_faithfulness_cost | 0.997 | 0.571 | 1.424 |
| default | 1.020 | 0.574 | 1.467 |
| lambda_zero | 1.022 | 0.575 | 1.469 |
| high_cost_pressure | 1.002 | 0.520 | 1.484 |

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
