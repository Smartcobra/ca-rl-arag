# CA-RL-ARAG — Milestone 2 Baseline & RL Environment

Cost-Aware Reinforcement Learning for Agentic RAG (V1 implementation).

This package delivers the Milestone 2 checklist from the research roadmap:

- Baseline RAG (`src/rag_baseline.py`)
- Agentic RAG loop with frozen actions `{retrieve, rewrite, rerank, verify, stop}` (`src/agentic_rag.py`)
- Gymnasium RL environment (`src/rag_env.py`)
- Explicit multi-component reward + ablation presets (`src/rewards.py`, `configs/reward_weights.yaml`)
- Dataset slices for **HotpotQA + single-hop** (NQ preferred; TriviaQA / SQuAD fallbacks — Scope Memo V2 §7)
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
```

**Ranking preflight.** `run_pilot.py` and `run_reward_ablation.py` (default config) refuse to load Qwen unless the **on-disk** eval file is 300 (150 Hotpot + 150 single-hop: NQ, TriviaQA, or SQuAD) **and** `corpus.jsonl` has at least **50,000** passages **and** there are no leftover NQ answer-anchors. `--limit` does not skip this — a 50-question run on a 2k corpus is still the wrong experiment. `--run-env-check` only rolls a few Gym episodes; it is not the data check. Bypass with `--skip-data-check` for synthetic/extractive debug. `configs/extractive.yaml` does not set the 50k floor. The committed ranking snapshot (`d456d26`) used Tevatron/wikipedia-nq.

**Full guide** (why each command, datasets, evaluation matrix, outputs): [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md).  
**Results explained** (pilot tables, how to read metrics/trajectories, ablations): [`docs/RESULTS.md`](docs/RESULTS.md).

## Pilot results (snapshot)

**Run this section describes:** 80k-passage Qwen ranking pilot, commit `d456d26` (2026-08-27). Source: `results/metrics/pilot_summary_default.json` — Qwen/Qwen2.5-3B-Instruct, `default` reward, full eval (`limit: null`). Slice: **300 examples (150 Hotpot + 150 NQ)**. Tevatron/wikipedia-nq loaded (`nq_corpus: dpr_wikipedia_w100`, `n_nq_anchor: 0`, 1,450 wiki golds / 847 eval gold articles). Corpus **80,000** passages (2,086 Hotpot slice + 1,450 DPR golds + 1,509 DPR negatives + 74,955 unused-Hotpot distractors). BM25 gold recall@5: Hotpot **0.927** (11 misses), NQ **0.587** (62 misses). Lexical NLI. `force_yes_no: true`, `allow_abstain: true`. The leaked-NQ write-up (148/150) is a different run: [`docs/NQ_MAX_TOOLS_ANALYSIS.md`](docs/NQ_MAX_TOOLS_ANALYSIS.md).

**Comparison in one line:** Hotpot is 59 / 56 / 61. NQ is 41 / 41 / 44. Extra tools move a few answers, but spend is 4.3× on `max_tools`, so reward still ranks **naive > rule > max_tools**. Single-hop is a ranking split now (~27–29% EM), not an answer-anchor ceiling.

Read **Hotpot and NQ**, not Overall.

### HotpotQA (n=150; ranking split)

| Policy | EM | F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.393 | 0.445 | 59/150 | 29 | 0.193 | 1.59e-4 | **0.654** |
| rule_based | 0.373 | 0.438 | 56/150 | 25 | 0.167 | 4.81e-4 | 0.590 |
| max_tools | **0.407** | **0.485** | **61/150** | 23 | 0.153 | 7.38e-4 | 0.587 |

Versus the leaked-NQ 80k run: Hotpot stays in the 56–61 band. `max_tools` vs naive is **3 recoveries / 1 regression** (net +2). That is still a handful of items — not a policy win.

### Natural Questions (n=150; ranking split — not saturated)

| Policy | EM | F1 | n_correct | n_abstained | abstain | mean $ | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.273 | 0.348 | 41/150 | 16 | 0.107 | 1.84e-4 | **0.542** |
| rule_based | 0.273 | 0.352 | 41/150 | 15 | 0.100 | 5.27e-4 | 0.504 |
| max_tools | **0.293** | **0.358** | **44/150** | 18 | 0.120 | 7.55e-4 | 0.464 |

Rule vs naive is **1 recovery / 1 regression** (tied). Max vs naive is **5 / 2** (net +3). Blind extra retrieves buy three NQ hits and still lose on reward.

### Overall (mix-weighted; do not rank from this)

| Policy | EM | F1 | n_correct | n_abstained | mean $ | mean steps | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 0.333 | 0.397 | 100/300 | 45 | 1.72e-4 | 2.0 | **0.598** |
| rule_based | 0.323 | 0.395 | 97/300 | 40 | 5.04e-4 | 4.0 | 0.547 |
| max_tools | **0.350** | **0.422** | **105/300** | 41 | 7.47e-4 | 7.0 | 0.526 |

Overall EM dropped **0.68 → 0.33** vs leaked NQ because copy-the-anchor is gone.

### Cost and action mix

| Policy | retrieve | rewrite | rerank | verify | mean $ | latency | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag | 1.0 | 0.0 | 0.0 | 0.0 | 1.72e-4 | 1089 ms | 795 |
| rule_based | 1.0 | 0.0 | 1.0 | 1.0 | 5.04e-4 | 1724 ms | 1599 |
| max_tools | 3.0 | 1.0 | 1.0 | 1.0 | 7.47e-4 | 3387 ms | 2055 |

Latency is high because BM25 scores 80k passages per query.

### What the three policies show

- **Retrieval can fail now.** Hotpot recall@5 is 0.927 (11/150). NQ recall@5 is 0.587 (62/150). That was the point of the distractor pool plus real (non-anchor) golds.
- **Hotpot quality is still a few-hit race** (59 / 56 / 61). Extra retrieves recover 3 and lose 1; not enough to pay 4.3× $.
- **NQ can rank policies on quality, barely.** Max 44 vs naive/rule 41. Rule is tied with naive at 1 recovery / 1 regression.
- **Abstain** on Hotpot is 29 / 25 / 23 (~15–19%). NQ abstain is 16 / 15 / 18 (~10–12%).
- **Cost still decides reward:** naive 0.598 > rule 0.547 > max_tools 0.526 (Hotpot 0.654 > 0.590 > 0.587; NQ 0.542 > 0.504 > 0.464).

Milestone 3 takeaway: both splits are hard enough that extra tools can change a few answers. A learned controller must **select** when that is worth the cost; blindly using max tools still loses on reward.

### Reward-weight ablation (not a ranking table)

Fixed `rule_based` policy, **stratified 100** from the **same Tevatron-NQ** 300-file (50 Hotpot + 50 NQ, commit `d456d26`). EM/F1/$ stay flat (EM 0.34); only the scalar reward changes. Source: `results/metrics/reward_ablation_table.json`.

| Preset | overall reward | Hotpot reward | NQ reward |
|---|---:|---:|---:|
| correctness_only | 0.362 | 0.373 | 0.351 |
| correctness_grounding | 0.547 | 0.495 | 0.598 |
| correctness_faithfulness_cost | 0.519 | 0.467 | 0.570 |
| default | 0.518 | 0.476 | 0.561 |
| lambda_zero | 0.520 | 0.478 | 0.563 |
| high_cost_pressure | 0.456 | 0.417 | 0.495 |

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
├── notebooks/         # Colab 80k ranking run
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
