# NQ analysis — does Max-Tools perform worse, and why?

Source: `results/metrics/pilot_summary_default.json` and `results/trajectories/{baseline,rule_based,max_tools}_default.jsonl`.  
Setup: Qwen2.5-3B-Instruct, `force_yes_no: true`, tighter ABSTAIN prompt, BM25, lexical NLI, 150 Natural Questions eval items. Corpus uses **answer-anchor passages** (`The answer is {ans}`), so NQ is a ceiling, not a retrieval ranking split.

**Short answer:** on this run Max-Tools is **not worse on NQ quality**. EM, F1, and the three misses are identical to naive and rule-based. It is worse on **cost, latency, tokens, and reward** because it always fires unused tools. The extra retrieve/rewrite/rerank/verify do not flip answers; they add spend and sometimes shuffle distractors around an anchor that already contains the gold.

---

## 1. Per-dataset NQ table (all policies)

| Metric | naive_rag | rule_based | max_tools |
|---|---:|---:|---:|
| **EM** | 0.980 | 0.980 | 0.980 |
| **F1** | 0.991 | 0.991 | 0.991 |
| n_correct | 147/150 | 147/150 | 147/150 |
| n_abstained | 0 | 0 | 0 |
| Q_ans | 0.986 | 0.986 | 0.986 |
| **Q_ground** | 0.997 | **1.000** | **1.000** |
| Q_cal | 0.286 | 0.286 | 0.286 |
| **P_hall** | 0.003 | **0.000** | **0.000** |
| mean reward | **1.501** | 1.464 | 1.403 |
| mean $ | **1.45e-4** | 4.53e-4 (3.1×) | 6.85e-4 (4.7×) |
| latency | **500 ms** | 1021 ms | 1638 ms |
| tokens | **616** | 1260 | 1584 |
| retrieve | 1.0 | 1.0 | 3.0 |
| rewrite | 0.0 | 0.0 | 1.0 |
| rerank | 0.0 | 1.0 | 1.0 |
| verify | 0.0 | 1.0 | 1.0 |
| steps | 2.0 | 4.0 | 7.0 |

Quality columns are the same to three decimals except grounding / hallucination, which differ on **one** item (`nq_eval_72`, below). Reward falls only because spend rises.

Per-example EM/F1 vs naive:

| Comparison | Count |
|---|---:|
| Max EM worse than naive | **0** |
| Max EM better than naive | **0** |
| Max F1 worse than naive | **0** |
| Max reward worse than naive | **149 / 150** |

There are no NQ cases where Max-Tools gets a worse answer than naive. “Worse” on NQ means **unnecessary actions and cost**, not a wrong span.

---

## 2. The three NQ misses (shared by all policies)

These are the only EM = 0 rows. Predictions match across naive / rule / max.

| id | Question | Gold | Prediction | EM / F1 | Why it fails |
|---|---|---|---|---|---|
| `nq_eval_46` | where is lord's prayer found in bible | `in the Gospel of Luke` | `Gospel of Luke` | 0 / 0.86 | Span is right; gold wants the preposition `in …`. Metric, not tools. |
| `nq_eval_63` | where does route 66 start on the west coast | `in Santa Monica` | `Santa Monica` | 0 / 0.80 | Same span-vs-`in …` mismatch. |
| `nq_eval_72` | is there a name for the at symbol | `commercial at` / `at symbol` / `at sign` | `no` | 0 / 0.00 | **Yes/no force false positive.** Question starts with `is`, so all three policies must emit yes/no. Gold is a name. |

None of these are Max-Tools-specific. Extra tools do not recover them.

### `nq_eval_72` is the only NQ grounding/hallucination gap

| Policy | pred | Q_ground | P_hall | reward |
|---|---|---:|---:|---:|
| naive | `no` | 0.50 | 0.455 | **−0.336** |
| rule_based | `no` | 1.00 | 0.00 | 0.279 |
| max_tools | `no` | 1.00 | 0.00 | 0.218 |

Naive’s tiny NQ `P_hall` (0.003) and `Q_ground` (0.997 vs 1.0) are **this one row**. After rerank, rule/max keep the NQ anchor and the lexical verifier marks claim `no` as supported (the token appears in the pool). That is a verifier rubber-stamp, not a better answer. This is also the **only** NQ item where Max reward > naive reward (naive is punished for hall; max is not).

Fix for this miss is the yes/no detector (`is there a name for …` is not a comparison question), not more retrieves.

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

1. **Do not say Max-Tools is worse on NQ EM/F1 on this run.** It is tied (147/150, same three misses, same predictions).
2. **Max-Tools is worse on NQ as a cost policy:** 4.7× $, 3.3× latency, lower reward on 149/150. That is the intended cost-ceiling signal (Scope Memo: NQ teaches when *not* to over-retrieve).
3. **Mechanism is unused tools, not a bad verify call.** Duplicate retrieve, unconditional rewrite (148/150), rerank, and a verify that always supports. Evidence membership changes on 72 items but EM does not, because the answer-anchor stays.
4. **The three misses are not tool failures.** Two are `in X` vs `X` EM strictness. One is the yes/no head firing on `is there a name for the at symbol`.
5. **NQ still cannot rank policies** until the corpus is real Wikipedia/DPR passages. Until then, treat NQ as a **cost-overuse diagnostic**, not a quality table.

### What would actually make Max-Tools less bad on NQ

- `is_yes_no_question` no longer treats `is there a name for …` as yes/no (`src/generation/llm.py`).
- Keep published `max_tools_*` artifacts as the high-cost reference. A learned stopper should copy naive on this split.
