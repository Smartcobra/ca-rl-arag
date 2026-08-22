# Reward Design Note (Milestone 2)

## Aggregate objective (Scope Memo V2 §6.3)

\[
R = \alpha Q_{\mathrm{ans}} + \beta Q_{\mathrm{ground}} + \gamma Q_{\mathrm{cal}}
- \lambda(C_{\mathrm{tok}} + C_{\mathrm{ret}}) - \mu C_{\mathrm{lat}}
- P_{\mathrm{hall}} - P_{\mathrm{act}} - P_{\mathrm{bud}}
\]

Efficiency / unused-budget bonus is **quality-gated**: granted only if \(Q_{\mathrm{ans}} \ge\) `quality_gate` (default 0.5), following GRASP’s \(R_E\) pattern so cheap wrong answers are not rewarded.

## Why these default weights?

| Weight | Default | Justification |
|---|---|---|
| **α (answer)** | 1.0 | Primary scientific signal; aligns with Search-R1 / GRASP \(R_A\). Must dominate so cost terms cannot buy reward via abstention-only policies. |
| **β (grounding)** | 0.4 | Material but secondary. Enough to punish uncited answers without overwhelming EM/F1. Mirrors GRASP’s substantial \(R_R\) (α=0.7 there on a different scale). |
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

Additionally, Milestone 3 will sweep `pareto_sweep.lambda_cost` × `mu_latency` for quality–cost curves.

## Component definitions (implementation)

| Symbol | Implementation |
|---|---|
| \(Q_{\mathrm{ans}}\) | `0.5 * EM + 0.5 * token-F1` |
| \(Q_{\mathrm{ground}}\) | Claim–evidence support (NLI/lexical) + Hotpot gold-title recall when available |
| \(Q_{\mathrm{cal}}\) | Justified abstain reward / confident-wrong penalty |
| \(C_{\mathrm{tok}}, C_{\mathrm{ret}}\) | From FinOps price card via `CostTracker` |
| \(C_{\mathrm{lat}}\) | `μ * seconds * latency_unit_usd` |
| \(P_{\mathrm{hall}}\) | Unsupported + contradiction mass × hall weights |
| \(P_{\mathrm{act}}\) | `act_penalty * max(0, n_actions - 1)` |
| \(P_{\mathrm{bud}}\) | `bud_penalty` if budget violated |

All components are logged per episode for methodology/discussion writing.
