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

- The paper is rewritten as one argument. Figure 1 is `figures/pipeline.png`.
  Figure 2 is the existing frontier, still the frozen-controller index.
  Table learned is the 16-d policy on the 83,120-passage index. The two runs
  are both seed 42. They are not labeled as different seeds.
- Naive, rule, and max-tools have **not** been re-scored on the 83,120 index.
  The two-index sentence stays in Limitations until that Colab pilot lands
  (`HOW_TO_RUN.md` §4.10). No training. Then delete that sentence and the
  ceilings can sit in the learned table.
- The saturated-tanh result is one diagnostics paragraph. The 0.0068 softmax
  probe is no longer a results claim.
- Three fresh seeds (42, 43, 44) are still planned (`HOW_TO_RUN.md` §4.11).

## 5. Build hygiene

- `paper/paper.pdf` and `paper/figures/frontier.png` stay tracked. The PDF on
  disk predates the 2026-10-05 text edit. Rebuild it before submitting.
  `pdflatex` is not installed on the machine that made that edit.
- `paper.aux`, `paper.bbl`, `paper.blg`, `paper.log`, and `missfont.log` are
  gitignored. If they are still in the index, `git rm --cached` them. Keep the
  PDF.
- Run `bibtex`, not just `pdflatex`. The bibliography key is `gao2022tevatron`.
