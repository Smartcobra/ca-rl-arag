# Submission checklist

The submission artifact is the anonymous build, `paper-anon.pdf`.
Compile `paper-anon.tex`, not `paper.tex` and not `paper copy.tex`.

```bash
cd paper_v4
pdflatex paper-anon && bibtex paper-anon && pdflatex paper-anon && pdflatex paper-anon
pdflatex ReproducibilityChecklist
```

`paper.tex` still has the name in `\author`. The `submission` option
replaces that block with "Anonymous submission", drops `\affiliations`,
and blanks `\pdfinfo`. There is no acknowledgements section and no
repository URL. After the PDF exists:

```bash
pdffonts paper-anon.pdf
strings paper-anon.pdf | grep -i -E 'parsuram|yobytech|github'
```

The font list should be Type 1 or TrueType. The `strings` command
should print nothing. `paper.pdf` in this folder is the old build.
Do not send it.

This machine has no `pdflatex`, so the page count and the font list
have not been checked here.

## Page limit

AAAI-27 main track, as posted: 7 pages of content, and pages 8--9 are
references only. The ethical statement is an unnumbered section before
the bibliography, so it counts in those 7 pages. Acknowledgements stay
out. The reproducibility checklist is a separate upload
(`ReproducibilityChecklist.tex`). It is not `\input` into the paper,
because a checklist after the references would either eat a content
page or land on a page the call reserves for references.

FrontierIR 2027 is listed as a workshop venue. Its call is not posted,
and the 2026 call did not state an Ethics and Limitations rule. Re-read
that call when it appears before treating the main-track page rule as
the workshop rule. The 2026 workshop allowed appendices outside the
page count. AAAI-27 main track does not.

## What the paper now says

One index, 83,120 passages. Frozen rows are one-shot 100, rule 97,
max tools 105, six-step recipe 100 with the same 300 answers as
one-shot. The oracle that may also use max tools is 107. Seeds 42, 43,
and 44 at lambda 20 select rewrite, rewrite, retrieve, stop and score
100. The union with one-shot is 122, which is not the ceiling. The
branch on retrieval strength is 0 of 3. BM25 recall at five stays
0.927 and 0.587, labeled as the 80,000-passage snapshot.

## Checklist answers that are not yes

- Hyperparameter ranges tried during development are not listed. The
  final values and the greedy-reward selection rule are. Answer: partial.
- Preprocessing code and experiment code are not in an appendix, and
  the anonymous paper has no repository link. Answers: no, no, partial.
- Seeds 42, 43, and 44 and temperature 0 are in the paper. The kernel
  flags are not. Answer: partial.
- The paper names one NVIDIA T4 and Qwen2.5-3B-Instruct. It does not
  list memory, the operating system, or library versions. Answer: partial.
- No statistical test. Answer: no.
- The theoretical follow-ups are NA because the paper has no theorems.
