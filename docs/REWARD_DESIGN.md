# Reward Design Note (Milestone 2)

## Aggregate objective (Scope Memo V2 §6.3)

<p align="center">
<em>R</em> = α <em>Q</em><sub>ans</sub> + β <em>Q</em><sub>ground</sub> + γ <em>Q</em><sub>cal</sub>
− λ(<em>C</em><sub>tok</sub> + <em>C</em><sub>ret</sub>) − μ <em>C</em><sub>lat</sub>
− <em>P</em><sub>hall</sub> − <em>P</em><sub>act</sub> − <em>P</em><sub>bud</sub>
</p>

Efficiency / unused-budget bonus is **quality-gated**: granted only if <em>Q</em><sub>ans</sub> ≥ `quality_gate` (default 0.5), following GRASP’s <em>R</em><sub>E</sub> pattern so cheap wrong answers are not rewarded.

## Why these default weights?

| Weight | Default | Justification |
|---|---|---|
| **α (answer)** | 1.0 | Primary scientific signal; aligns with Search-R1 / GRASP <em>R</em><sub>A</sub>. Must dominate so cost terms cannot buy reward via abstention-only policies. |
| **β (grounding)** | 0.4 | Material but secondary. Enough to punish uncited answers without overwhelming EM/F1. Mirrors GRASP’s substantial <em>R</em><sub>R</sub> (α=0.7 there on a different scale). |
| **γ (calibration)** | 0.15 | Non-zero so cost pressure does not collapse into unjustified `abstain`. Smaller than β because abstention is an auxiliary behavior. |
| **λ (token+retrieval $)** | 2.0 | Scales FinOps dollars (typically ≪ 0.01 per episode on this price card) into the same numeric range as quality ∈ [0,1]. Tuned so ~1–2 extra retrieves are visible but not catastrophic. |
| **μ (latency)** | 0.5 | Latency is important for SLOs but usually secondary to $ in V1; keep lower than λ and sweep later. |
| **hall_weight** | 0.5 | Unsupported-claim penalty comparable to a partial F1 miss. |
| **verify_ignored_extra** | +0.15 | Stronger hallucination pressure when `verify` was available but unused — directly tests sub-question on verify value. |
| **act_penalty** | 0.02 | Anti-loop; deliberately ≪ cost of one useful retrieve so the agent is not punished for necessary tools. |
| **bud_penalty** | 1.0 | Hard fail signal when $ / tokens / latency / steps exceeded. |
| **quality_gate** | 0.5 | GRASP-style gate before efficiency credit. |
| **unused_budget_bonus** | 0.1 | Small bonus for saving budget *after* being correct enough. |

## Ablation plan (must run)

Configured in `configs/reward_weights.yaml` and executed by `scripts/run_reward_ablation.py`:

1. **correctness_only** — Search-R1-like outcome reward  
2. **correctness_grounding** — add faithfulness / grounding / hall  
3. **correctness_faithfulness_cost** — add cost terms (no calibration)  
4. **default** — full objective  
5. **lambda_zero** — isolate benefit of measured $ / latency  
6. **high_cost_pressure** — cheap operating point on the Pareto frontier  

Additionally, Milestone 3 will sweep `pareto_sweep.lambda_cost` × `mu_latency` for quality–cost curves. The first REINFORCE run used the **default** preset only; it is a train curve, not a new ablation.

**Trainer sanity (2026-09-13), not a rescore:** presets 1 and 5 were used as *training* objectives. `correctness_only` frozen eval left two steps (8 / 3 retrieve / 2 verify, EM 0.340). `lambda_zero` frozen eval stayed retrieve→stop. That isolates \(P_{\mathrm{act}}\) as the term that pins the ranking `learned` row to naive; λ is almost unused. Details: [`RESULTS.md`](RESULTS.md) §6.

## Component definitions (implementation)

| Symbol | Implementation |
|---|---|
| <em>Q</em><sub>ans</sub> | `0.5 * EM + 0.5 * token-F1` |
| <em>Q</em><sub>ground</sub> | Claim–evidence support (NLI/lexical) + Hotpot gold-title recall when available |
| <em>Q</em><sub>cal</sub> | See **What <em>Q</em><sub>cal</sub> scores** below. Implemented in `calibration_score` (`src/rewards.py`) |
| <em>C</em><sub>tok</sub>, <em>C</em><sub>ret</sub> | From FinOps price card via `CostTracker` |
| <em>C</em><sub>lat</sub> | `μ * seconds * latency_unit_usd` |
| <em>P</em><sub>hall</sub> | Unsupported + contradiction mass × hall weights |
| <em>P</em><sub>act</sub> | `act_penalty * max(0, n_actions - 1)` |
| <em>P</em><sub>bud</sub> | `bud_penalty` if budget violated |

All components are logged per episode for methodology/discussion writing.

## What <em>Q</em><sub>cal</sub> scores

Calibration is **not** answer correctness (<em>Q</em><sub>ans</sub> is EM/F1). It scores whether the policy’s confidence matches the evidence: refuse when retrieval is weak, answer when it is not.

Evidence is **weak** (justified abstain) only if any of these hold: no passages, mean retrieval `score` `< 3.0` (same cutoff the rule policy uses as “not enough to stop”), or `verify_out.label == "contradiction"`. Otherwise an abstain is treated as a lazy refuse. Using gold-wrong as “justified” is forbidden: that branch was a tautology (`not correct` is always true after the refused-solvable check) and taught always-abstain as easy reward, especially since <em>P</em><sub>hall</sub> is also zeroed on abstain.

| What happened | <em>Q</em><sub>cal</sub> |
|---|---|
| Answered correctly | `+0.3` |
| Abstained, evidence weak (empty **or** mean score `< 3.0` **or** verify=`contradiction`) | `+0.6` |
| Abstained, evidence usable (lazy refuse) | `-0.2` |
| Answered wrong, no evidence | `-0.2` |
| Answered wrong, with evidence (confident hallucination) | `-0.4` |
| Abstained when the prediction already matched gold | `-0.5` |

## Impact on the ranking snapshot (2026-09-04)

The Tevatron-NQ 80k pilot was rescored with this rule. EM/F1/$ did not move (frozen policies). Mean <em>Q</em><sub>cal</sub> dropped (overall −0.017 → −0.137 on naive) because lazy abstains are no longer +0.6. Mean reward dropped in lockstep (naive 0.598 → 0.580); ranking is still naive > rule > max_tools. The `default` REINFORCE train curve and the 300-eval `learned` row (tied with naive) used this same fixed rule. The 2026-09-13 free-cost trains reused it for `correctness_only` / `lambda_zero` (trainer sanity, not a new ranking row). Details: [`RESULTS.md`](RESULTS.md).
