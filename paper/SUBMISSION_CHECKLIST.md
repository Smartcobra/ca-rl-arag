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

- The paper describes the **v2** run: 261 parameters, a 10-d observation, 100
  training questions, and an 80,000-passage index. The repository has since moved
  to a 16-d observation, 357 parameters, 500 training questions, and an
  83,120-passage index, and the retrained policy **branches** where the paper says
  it does not. Decide deliberately whether this submission is the v2 story or the
  v3 story; do not mix rows from the two. See `docs/RESULTS.md` §6.
- One index per table (`docs/IMPLEMENTATION_DECISIONS.md`, 2026-10-01). The
  frozen baselines have **not** been re-scored on the 83,120 index.
- Seeds: a single run at seed 42. See `docs/IMPLEMENTATION_DECISIONS.md`,
  2026-10-03, for the three-seed plan and what it costs.
- `python scripts/verify_softmax_probe.py` re-checks the 0.0068 figure in
  Section "Constant Recipes Leave a Per-Question Gap".

## 5. Build hygiene

- `paper.pdf`, `paper.aux`, `paper.bbl`, `paper.blg`, and `paper.log` are tracked
  and currently **stale**: they predate the 2026-10-03 limitations edit. Rebuild
  and commit before submitting, or the committed PDF will not match the source.
- Run `bibtex`, not just `pdflatex`. The `gao2022tevatron` key was renamed on
  2026-10-03; `paper.bbl` was hand-synced to match, but a real rebuild is the
  only way to be sure the bibliography is consistent.
