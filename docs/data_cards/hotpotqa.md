# Data Card — HotpotQA (V1 slice)

## Identity
- **Dataset:** HotpotQA (Yang et al.)
- **Config used:** `distractor`
- **Role (Scope Memo V2):** multi-hop train/eval mix; supporting-fact titles enable grounded-reading ablations

## Motivation
Multi-hop questions force the agent to decide when another `retrieve` / `rewrite` is worth its cost versus `stop` / `verify`.

## Composition (this repo slice)
- Built by `scripts/prepare_data.py`
- Train/eval counts: see `data/processed/slice_meta.json`
- Corpus passages: Wikipedia paragraphs from Hotpot contexts in the selected rows (synthetic mode uses a closed fact corpus)

## Preprocessing
- Each context paragraph → one passage `{passage_id, title, text}`
- Gold supporting titles retained for \(Q_{\mathrm{ground}}\) / GRASP-style grounded reading
- Answers stored as short strings for EM/F1

## Splits
- Train slice for policy debugging / later RL
- Eval slice for pilot metrics (locked: **150** eval examples in the 300 mixed file)

## Labeling / gold
- Answer string + supporting fact titles (distractor setting)

## Known limitations
- Slice is 150 eval / 60 train here; still not for claiming SOTA.
- Global BM25 index mixes passages across examples (realistic open-corpus setup; harder than per-example context). On the current 80k index, Hotpot BM25 R@5 is **0.927** (11 miss@5).
- Current ranking mix is Hotpot + **NQ** (`d456d26`, reward rescored 2026-09-04). Hotpot EM is 59 / 56 / 61 (naive / rule / max); reward is 0.631 / 0.570 / 0.569.
- Synthetic fallback is not Hotpot distribution

## Ethical / license notes
- Use upstream HotpotQA license/terms when downloading from HuggingFace
- Do not redistribute large raw dumps beyond project needs
