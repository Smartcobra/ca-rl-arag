# Submission checklist

Work through this before sending the paper anywhere. Items are ordered by how
expensive they are to discover late.

## 1. Is the venue double blind? — UNRESOLVED

**No workshop is named anywhere in this repo yet.** The paper builds against
`aaai2027.sty`, which is the AAAI main-conference template; that is a format
choice and says nothing about the review policy of the workshop we target.

Decide this before the author block matters, because it decides which file is
the submission artifact:

| Review policy | Build | Author block |
|---|---|---|
| Double blind | `paper-anon.tex` | "Anonymous submission" |
| Single blind / open | `paper.tex` | Named |

Check the call for papers for the specific workshop, not the host conference.
Workshops frequently differ from their parent conference in both directions.
Also check whether the venue bars or allows a preprint, which is a separate
question from blinding and has caught people out.

## 2. Anonymous build

`paper-anon.tex` exists and passes the template's `submission` option. Build it
with:

```bash
cd paper
pdflatex paper-anon && bibtex paper-anon && pdflatex paper-anon && pdflatex paper-anon
```

**Not yet compiled.** No TeX installation was available when this file was
written, so the wrapper is untested. Build it once and confirm by eye before
relying on it.

What the option does: replaces `\author` with "Anonymous submission", drops
`\affiliations`, and blanks `\pdfinfo` so the title does not ride along in the
PDF metadata. What it does **not** do is scrub prose. As of this writing the
only identifying content in `paper.tex` is the author block itself (lines 19-21)
— there is no acknowledgements section, no repository URL, and no self-citation
— so the option alone is currently sufficient. Re-check after any edit that adds:

- an acknowledgements or funding section
- a link to code, data, or a personal or institutional page
- a phrase like "our previous work" with a non-anonymous citation
- an author name inside a figure, a plot legend, or a screenshot

Verify the built PDF two ways, because the visible page and the file metadata
fail independently:

```bash
pdffonts paper-anon.pdf                    # sanity: it built
strings paper-anon.pdf | grep -i -E 'parsuram|yobytech'   # must print nothing
```

## 3. Authors and affiliation — deferred on purpose

`paper.tex` currently carries a real name and email. That is fine for a named
submission and fatal for a blind one. Leave it as is; the anonymous build routes
around it without edits. Fix the final form of the author list and affiliation
when the venue is settled.

## 4. Numbers

- The paper now has both indexes, in separate tables. Table frozen, Table learned,
  and `figures/frontier.png` are the 80,000-passage dollar sweep. The later table
  is the 83,120-passage index: the 15-d validation pick (λ=0 at 103/300, one
  six-step recipe; λ=20 and λ=80 one-shot) and two seed-42 draws of the 16-d
  z-scored observation (103/300 with 16 sequences, then 104/300 without that
  controller). Do not add a row from one table to the other.
- One index per table (`docs/IMPLEMENTATION_DECISIONS.md`, 2026-10-01). Naive,
  rule, and max-tools have **not** been re-scored on the 83,120 index, so the
  frontier figure has not been rebuilt. That is the remaining Colab pilot
  (`HOW_TO_RUN.md` §4.10). No training.
- Seeds: the learned rows are seed 42. The two 16-d draws are the same seed run
  twice. Three fresh seeds (42, 43, 44) are planned if units allow
  (`HOW_TO_RUN.md` §4.11).
- `python scripts/verify_softmax_probe.py` re-checks the 0.0068 figure in
  Section "Constant Recipes Leave a Per-Question Gap".

## 5. Build hygiene

- `paper/paper.pdf` and `paper/figures/frontier.png` stay tracked. The PDF on
  disk predates the 2026-10-05 text edit. Rebuild it before submitting.
  `pdflatex` is not installed on the machine that made that edit.
- `paper.aux`, `paper.bbl`, `paper.blg`, `paper.log`, and `missfont.log` are
  gitignored. If they are still in the index, `git rm --cached` them. Keep the
  PDF.
- Run `bibtex`, not just `pdflatex`. The bibliography key is `gao2022tevatron`.
