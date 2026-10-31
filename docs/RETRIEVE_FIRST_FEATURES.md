# Retrieve first, then look at the score

This note is the change list for one new training variant. It is written for someone who has not trained the policy before. The three λ = 20 exams are scored below. `paper_v4/paper.tex` still describes the earlier 16-number run. Do not paste these rows into the paper from this note.

## What is wrong today

An episode is one question. At each step the policy sees a list of 16 numbers and picks one action: retrieve, rewrite, rerank, verify, or stop.

At step 0 that list is the same for every question.

- The four BM25 score slots are 0, because nothing has been retrieved.
- The list does not contain the question.
- The list does not contain a draft answer, because the generator has not run.

The trained policy uses that moment to call rewrite, then rewrite again, then retrieve, then stop. Both rewrites are spent while the score slots are still 0. The second decision never sees a BM25 score. Our own claim is that a controller should read the evidence before it spends a rewrite. This variant is the direct test of that claim.

## What this variant does

Three additions, in one run.

1. The first action is always retrieve. After that, the policy chooses rewrite, retrieve again, or stop, and the score slots are real BM25 numbers.
2. A few question features are appended to the observation, so the decision after the retrieve can depend on the question. There is still no dataset id.
3. Two draft features are appended. They stay 0 until a draft exists. They tell the policy whether that draft abstained, and how much of it overlaps the top passage.

The parameter cap stays 720. Hidden width stays 16.

### The first action and the question features do different jobs

Forcing the first action to be retrieve means step 0 is not a learned choice. The question features are in the vector at step 0, and the mask ignores them there. They matter on the next step, together with the BM25 scores.

That is the run to train. A later run can turn the force flag off. That later run is the one where step 0 itself can change with the question. It is not this test. If step 0 is allowed to rewrite, the policy can go back to spending rewrites before it has a score.

### The draft features are not visible on the second action

Retrieve does not call the generator. After the forced retrieve, the draft is still empty, so both draft features are 0. The second action sees the question features and the BM25 scores. It does not see an answer.

The generator runs in two places.

- `verify`, if there is no draft yet. The next step can see the draft features.
- `stop`, which ends the episode. The policy never acts on the draft created at stop.

So the draft features are a signal for the step after verify: the draft abstained, or it barely overlaps the top passage, so rewrite or retrieve again instead of stopping. Do not add an extra generate call to fill these features. A generate call is on the price card. These two numbers are string checks on a draft that already exists.

## The new vector

Keep the current 16 numbers in the same order. `legal_mask` reads indices 1, 2, and 6–9. The score tests read indices 0, 10, 11, and 15. Append only.

| Index | Feature | Value when it is missing |
| --- | --- | --- |
| 0–15 | unchanged | same as today |
| 16 | question length | 0 |
| 17 | question word: who | 0 |
| 18 | question word: what | 0 |
| 19 | question word: when | 0 |
| 20 | question word: which | 0 |
| 21 | question word: yes/no | 0 |
| 22 | capitalized entity count | 0 |
| 23 | draft abstained | 0 |
| 24 | draft overlap with the top passage | 0 |

`OBS_DIM` goes from 16 to 25.

Build the question features from `query`, the original question. Do not use `current_query`. A rewrite changes `current_query`, and the feature would then move because of an action the policy already took.

**Length.** Count words. Divide by 32 and clip to 1. The other non-score channels already live in `[0, 1]`. A question of 40 words saturates at 1. Do not z-score this on the training slice.

**Question word.** One-hot, five slots. Read the first word.

- `who` or `whom` sets index 17.
- `what` sets 18.
- `when` sets 19.
- `which` sets 20.
- Otherwise, if `is_yes_no_question` in `src/generation/llm.py` is true, set index 21. Use that function. The generator already uses it for yes/no answers.
- `where`, `why`, `how`, and everything else stay all zeros. Do not add an "other" slot that is 1 on those questions. Do not add a dataset slot.

**Capitalized entities.** Split the raw question on words. Skip the first word, so "Who" and "What" are not counted. Count the remaining words that start with a capital letter. "Who is Barack Obama?" counts 2. Divide by 5 and clip to 1. This is not a named-entity model.

**Draft abstained.** 1 when the draft string is `ABSTAIN`. An empty draft is 0. Do not read `stop_mode` for this. `stop_mode` is set on the stop action, and the episode is already over.

**Overlap.** Use `tokenize` from `src/utils.py` on the full draft and on the top passage (`title` plus `text` of `evidence[0]`). The value is the fraction of draft tokens that also appear in that passage. An empty draft, an abstain, or an empty passage is 0. Compute this from the full draft, before the 200-character truncate used in the logged observation. The vector gets this one number. It does not get the passage text.

Do not read `obs["dataset"]`. `tests/test_learned_policy.py` already checks that adding a dataset name leaves the vector unchanged. Keep that check and extend it to the new channels.

## Parameter count

The network is one hidden layer of width 16, then 5 actions.

```
(25 * 16 + 16) + (16 * 5 + 5) = 416 + 85 = 501
```

501 is under the cap of 720. Leave `MAX_PARAM_COUNT` at 720. Leave the default hidden width at 16.

Width 32 does not fit anymore:

```
(25 * 32 + 32) + (32 * 5 + 5) = 832 + 165 = 997
```

The largest width that still fits is 23, which is 718 parameters. `PolicyMLP(32)` must raise. The test that currently expects a 709-parameter network at width 32 has to expect that error. Do not raise the cap to make width 32 legal.

## Where the forced retrieve lives

Put it in `legal_mask` in `src/policies/learned.py`, behind a config flag:

```yaml
agent:
  force_first_retrieve: true
```

When the flag is on, the retrieve count is 0, and this is not the last step, the mask allows only retrieve. The last-step rule stays first: one step left still means stop.

The trainer already calls `policy.act(..., cfg=cfg)`, and that call already applies the mask. A masked action gets no probability, so REINFORCE does not treat a blocked rewrite as if the policy had chosen retrieve.

Do not silently replace the action inside `AgenticRAGEnv.step`. The trainer only fixes the log-probability when the environment overrides the policy, and that path is for the budget stop and the last step.

`naive_rag`, `rule_based`, and `max_tools` do not use this mask. Leave them alone. `rule_based` already retrieves first.

## Files to change

### Code

| File | Change |
| --- | --- |
| `src/rag_env.py` | Set `OBS_DIM = 25`. Append the nine features in `vectorize_structured_obs`. Add small helpers for the question word, the entity count, and the overlap. Missing question, missing draft, and missing passage stay 0. Do not move indices 0–15. Do not read a dataset id. |
| `src/agentic_rag.py` | `AgentState.observation` already has `query` and a truncated `draft_answer`. Add the full draft and the top passage text (title plus text), or add the two draft floats computed before the truncate. The vectorizer needs those. Do not put the passage string into the 25 numbers. |
| `src/policies/learned.py` | Update the layout comment at the top. In `legal_mask`, honor `agent.force_first_retrieve`. Update the docstring that says 357 parameters. Leave indices 1, 2, and 6–9 alone. Leave `MAX_PARAM_COUNT` at 720. The load error can mention that a 16-d checkpoint will not load into this head. |

`scripts/train_policy.py` already passes `cfg` into the mask. It does not need a new training rule. The new checkpoint path comes from the new config.

`scripts/smoke_test.py` and `tests/test_train_resume.py` use `OBS_DIM`. They follow the constant. No edit unless a run shows a hardcoded 16.

### Config

Copy `configs/frontier_v3.yaml` to `configs/retrieve_first.yaml`. Change only:

- `agent.force_first_retrieve: true`
- `policy.learned.checkpoint` to `results/checkpoints/learned_policy_retrieve_first.pt`

Leave `configs/frontier_v3.yaml` as it is. That file is the published run. Training this variant must not overwrite `learned_policy_frontier_v3.pt` or the v4 checkpoints. `LearnedPolicy.load` already refuses a checkpoint whose `obs_dim` is not the live one. A 16-d file stays on disk and does not load into this head.

### Tests

| File | Change |
| --- | --- |
| `tests/test_learned_policy.py` | `OBS_DIM` is 25. Width 16 has 501 parameters and is under 720. Width 32 raises. Add a mask test: with the flag on and a zero retrieve count, only retrieve is legal; with the flag off, retrieve and rewrite are both legal when there is no evidence. Add a vector test: "Who is Barack Obama?" sets who and a non-zero entity count; a yes/no question sets only the yes/no slot; a tagged `dataset` key still does not change the vector; an `ABSTAIN` draft sets the abstain slot and overlap 0; a draft that shares tokens with the top passage sets overlap above 0; an empty draft leaves both draft slots at 0. |
| `tests/test_score_features.py` | The stacked shape is `(300, 25)`. Score indices 0, 10, 11, and 15 stay. The trajectory helper does not pass a question or a draft, so the nine new channels are 0 on those rows. The score standard-deviation checks stay as they are. |

### Comment only

`scripts/verify_softmax_probe.py` replays a frozen 10-d vector and pads to `OBS_DIM`. Leave that replay alone. Update the note that says the live observation is 16-d. After this change the live observation is 25-d.

### Do not change

- `MAX_PARAM_COUNT` (720)
- the reward, the price card, and the BM25 z-score file
- `src/policies/__init__.py` (the frozen controllers)
- `paper_v4/paper.tex`. The exams are scored in the section below. The tex still describes the 16-number run.
- the old result write-ups in `docs/RESULTS.md` and `docs/EXPERIMENT_LOG.md`, except the dated retrieve-first entry. Those earlier sections describe the 16-d run.

## How to tell the code is right before a GPU run

From `agentic_rag_rl`:

```bash
python tests/test_learned_policy.py
python tests/test_score_features.py
```

Then one training command, pointed at the new config, writes `results/checkpoints/learned_policy_retrieve_first.pt`. It must not replace the frontier v3 or v4 checkpoints.

On the exam trajectories, check three facts.

1. The first action is retrieve on every question.
2. The second action is chosen while `mean_score` is a real BM25 value, not 0.
3. The branch count uses the same rule as the paper: the second action is rewrite on at least 15 questions and retrieve on at least 15, and one threshold on the mean score matches at least 250 of those questions. Report the count either way. A run that still picks one string after it has seen the score is a result.

Also check that the nine new channels are not constant on the training questions. Length, the question-word slots, and the entity count should move at step 0. The two draft slots should stay 0 until a draft exists, and should move on steps after verify.

## What the three exams did

Trained at λ = 20, seeds 42, 43, and 44, on `configs/retrieve_first.yaml`. Same 500 / 180 / 300 split. The kept checkpoint is the first epoch with the highest greedy reward on the 180. The exam files are `learned_frontier_lambda20_rfs42`, `rfs43`, and `rfs44`.

The first action is retrieve on 300/300 in every file. **The branch appears in 0 of 3 seeds.** Report the three correct counts. Do not report a mean and a standard deviation.

| Seed | Kept epoch | Greedy reward on the 180 | Hotpot | NQ | Correct | Dollars | Steps | Sequences | Second action | Branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 17 | 0.608 | 60/150 | 43/150 | 103/300 | 4.60×10⁻⁴ | 6.51 | 15 | rewrite 230, retrieve 66, verify 4 | no, 236/296 |
| 43 | 2 | 0.590 | 60/150 | 40/150 | 100/300 | 1.72×10⁻⁴ | 2 | 1 | stop 300 | no |
| 44 | 19 | 0.596 | 60/150 | 43/150 | 103/300 | 5.62×10⁻⁴ | 8 | 1 | verify 300 | no |

Seed 42’s common path is retrieve, rewrite, rewrite, retrieve, retrieve, stop (164/300). Rewrite and retrieve both clear 15 as the second action, and the best mean-score threshold matches 236 of 296. That is under 250.

Seed 43 is retrieve, then stop, on 300/300. Predictions and the dollar bill match this index’s one-shot (`baseline_default_v4`, 100/300, Hotpot 60 / NQ 40, 1.72×10⁻⁴).

Seed 44 is one sequence on 300/300: retrieve, verify, verify, rewrite, rewrite, retrieve, retrieve, stop. The second action is verify, so the rewrite/retrieve counts are 0 and 0.

Against that one-shot file: seed 42 is +7 / −4 (Hotpot +1 / −1, NQ +6 / −3), union 107/300. Seed 43 is +0 / −0. Seed 44 is +9 / −6 (Hotpot +1 / −1, NQ +8 / −5), union 109/300.

Figures: `results/figs/retrieve_first_exams_table.png`, `retrieve_first_vs_oneshot_table.png`, `retrieve_first_em_usd.png`, `retrieve_first_second_action.png`. Rebuild them with `python scripts/plot_retrieve_first_exams.py`.
