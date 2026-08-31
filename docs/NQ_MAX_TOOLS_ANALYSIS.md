# NQ analysis — does Max-Tools perform worse, and why?

**Run this doc describes:** tiny-corpus Qwen 300-eval, **2,276 passages**, leaked-anchor NQ **148/150** (EM 0.987). Metrics/trajectories: commit `34e6585` (2026-08-22); this write-up: `6343b54`. Setup: Qwen2.5-3B-Instruct, `force_yes_no: true`, tighter ABSTAIN prompt, BM25, lexical NLI, 150 NQ items, answer-anchor passages (`The answer is {ans}`). NQ is a ceiling, not a retrieval ranking split.

**Current ranking snapshot is not this file.** `results/metrics/pilot_summary_default.json` is commit `d456d26` (2026-08-27): 150 Hotpot + 150 **NQ** (Tevatron/wikipedia-nq), overall EM ~0.33. Do not re-read those JSON paths for the tables below. The SQuAD fallback `e8a4423` and leaked-NQ 80k snapshot `2417c43` (NQ 146/150) are also historical.

**Short answer (this leaked-NQ run only):** Max-Tools is **not worse on NQ quality**. EM/F1 match naive and rule (148/150 after the yes/no detector fix). It is worse on **cost, latency, tokens, and reward** because it always fires unused tools. The remaining two misses are the same `in …` span mismatch on all policies.

---

## 1. Per-dataset NQ table (all policies)

| Metric | naive_rag | rule_based | max_tools |
|---|---:|---:|---:|
| **EM** | 0.987 | 0.987 | 0.987 |
| **F1** | 0.998 | 0.998 | 0.998 |
| n_correct | 148/150 | 148/150 | 148/150 |
| n_abstained | 0 | 0 | 0 |
| Q_ans | 0.986 | 0.986 | 0.986 |
| **Q_ground** | 1.000 | 1.000 | 1.000 |
| Q_cal | 0.291 | 0.291 | 0.291 |
| **P_hall** | 0.000 | 0.000 | 0.000 |
| mean reward | **1.514** | 1.472 | 1.411 |
| mean $ | **1.45e-4** | 4.54e-4 (3.1×) | 6.86e-4 (4.7×) |
| latency | **492 ms** | 1031 ms | 1632 ms |
| tokens | **616** | 1260 | 1584 |
| retrieve | 1.0 | 1.0 | 3.0 |
| rewrite | 0.0 | 0.0 | 1.0 |
| rerank | 0.0 | 1.0 | 1.0 |
| verify | 0.0 | 1.0 | 1.0 |
| steps | 2.0 | 4.0 | 7.0 |

Quality columns match to three decimals. Reward falls only because spend rises. The old `nq_eval_72` grounding/hall gap is gone: all three now predict `commercial at` (EM 1).

Per-example EM/F1 vs naive:

| Comparison | Count |
|---|---:|
| Max EM worse than naive | **0** |
| Max EM better than naive | **0** |
| Max F1 worse than naive | **0** |
| Max reward worse than naive | **150 / 150** |

There are no NQ cases where Max-Tools gets a worse answer than naive. “Worse” on NQ means **unnecessary actions and cost**, not a wrong span.

---

## 2. The two remaining NQ misses (shared by all policies)

These are the only EM = 0 rows. Predictions match across naive / rule / max.

| id | Question | Gold | Prediction | EM / F1 | Why it fails |
|---|---|---|---|---|---|
| `nq_eval_46` | where is lord's prayer found in bible | `in the Gospel of Luke` | `Gospel of Luke` | 0 / 0.86 | Span is right; gold wants the preposition `in …`. Metric, not tools. |
| `nq_eval_63` | where does route 66 start on the west coast | `in Santa Monica` | `Santa Monica` | 0 / 0.80 | Same span-vs-`in …` mismatch. |

Neither is Max-Tools-specific. Extra tools do not recover them.

### `nq_eval_72` is now a hit (yes/no detector)

Previously all three predicted `no` because `is there a name for the at symbol` was treated as yes/no. After tightening `is_yes_no_question`, all three predict `commercial at` (EM 1, Q_ground 1, P_hall 0). That is the NQ 147 → 148 jump. It is not a tool effect.

---

## 3. What Max-Tools actually does on NQ

Frozen script on every item: `retrieve → retrieve → rewrite → retrieve → rerank → verify → stop`.

| Behavior (150 NQ) | Count | Meaning |
|---|---:|---|
| 2nd retrieve = 1st retrieve (same top-5) | **150 / 150** | Pure no-op. Same query, paid twice. |
| Rewrite changes the query | **148 / 150** | Almost always. |
| 3rd retrieve changes top-5 | **57 / 150** | Rewrite can move BM25 hits. |
| Final evidence **identical** to naive | 59 | Extra tools did nothing to context. |
| Same 5 docs, reordered | 19 | Rerank shuffle. |
| Membership changed vs naive | **72** | New/dropped passages. |
| Of those 72, EM still 1 | **70** | Anchor stay in top-5; gold still copied. |
| Verify label `support` | **150 / 150** | Verify never rejects an NQ answer. |

Mean Max NQ $ by action:

| Action | mean $ | Necessary on NQ? |
|---|---:|---|
| retrieve ×3 | 1.50e-4 | First retrieve is enough. 2nd is a no-op. 3rd is optional noise. |
| rerank | 2.79e-4 | Largest line item. Does not change EM. |
| generate / stop | 1.93e-4 | Required. |
| rewrite | 5.3e-5 | Changes query 148 times; does not add EM. |
| verify | 1.0e-5 | Always `support`. |

So Max-Tools is taking **unnecessary actions** on essentially every NQ example. It is not making a careful verification/retrieval decision that then hurts the span. The controller never decides; the script always spends.

---

## 4. Trajectory notes: noisy context that did not flip EM

Because the NQ index is answer-anchored, a bad rewrite can inject distractors **and still copy the gold** from `NQ anchor for nq_eval_*`.

**Rewrite drifts the entity (still EM = 1)** — `nq_eval_5`:

- Question: `when did the isle of wight become an island`
- Max rewrite: `when did the isle of man become an island`
- Naive / max both predict `During the last Ice Age.`
- The Wight **anchor stays in slot 1**, so the wrong hop does not matter.

**Rewrite leaks the answer into the query, then pulls a distractor** — `nq_eval_63` (already an EM miss for everyone):

- Rewrite: `where does route 66 start on the west coast Santa Monica`
- Retrieve 3 drops `Hawks Nest, West Virginia` and adds `Santa's Workshop (Colorado)` (token overlap on “Santa”).
- Prediction stays `Santa Monica` (same as naive). Noisy context, no extra miss.

**Membership change while still correct** — 70 items, e.g. `nq_eval_0` (moon landing): rewrite `when did the last person land on the moon` swaps some other NQ anchors / `Moon (visual novel)` in/out. Gold `14 December 1972 UTC` is still on the item’s own anchor.

This is why NQ cannot show “Max-Tools hurts quality”: the gold string is planted in the corpus. Extra tools mostly rearrange neighbors.

---

## 5. Verification / retrieval decisions

Max-Tools does **not** use verify to stop, abstain, or pick a different answer. `stop` always regenerates. On all 150 NQ items verify is `support`.

Retrieval is not a decision either: retrieve is called three times regardless of scores. The second call never changes hits. The third call is a rewrite-conditioned BM25 merge, not “retrieve because evidence was weak.”

Rule-based on NQ is the same quality at 3.1× cost (one rerank + one verify, no rewrite). It is a cheaper version of the same unnecessary-tool pattern, without the duplicate retrieve.

---

## 6. Conclusions

1. **Do not say Max-Tools is worse on NQ EM/F1 on this run.** It is tied (148/150 here; leaked-NQ 80k snapshot `2417c43` is 146/150, still tied). Same two `in …` misses, same predictions. The current Tevatron-NQ ranking run (`d456d26`) is a different corpus — max vs naive there is 5 recoveries / 2 regressions (41 → 44), still not an EM win that pays 4.3× $. The SQuAD fallback (`e8a4423`) was 8 / 8.
2. **Max-Tools is worse on NQ as a cost policy:** 4.7× $, 3.3× latency, lower reward on **150/150**. That is the intended cost-ceiling signal (Scope Memo: NQ teaches when *not* to over-retrieve).
3. **Mechanism is unused tools, not a bad verify call.** Duplicate retrieve, unconditional rewrite (148/150), rerank, and a verify that always supports. Evidence membership changes on 72 items but EM does not, because the answer-anchor stays.
4. **The two remaining misses are not tool failures.** Both are `in X` vs `X` EM strictness. The old at-symbol miss is fixed by the yes/no detector.
5. **Leaked-anchor NQ cannot rank policies.** Treat this write-up as a **cost-overuse diagnostic** on a planted gold. The current ranking table is Hotpot + Tevatron NQ (`d456d26`, [`RESULTS.md`](RESULTS.md)).

### What would actually make Max-Tools less bad on NQ

- `is_yes_no_question` no longer treats `is there a name for …` as yes/no (`src/generation/llm.py`).
- Keep published `max_tools_*` artifacts as the high-cost reference. A learned stopper should copy naive on this split.
