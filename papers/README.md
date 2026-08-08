# Papers

**If you came here to read the papers, they are in [`pdf/`](pdf/).**

| Paper | Question it answers |
|---|---|
| [`pdf/1-alphaaudit.pdf`](pdf/1-alphaaudit.pdf) | Is the signal real, or is it leakage and multiple testing? |
| [`pdf/2-regimestress.pdf`](pdf/2-regimestress.pdf) | When does a surviving strategy break? |
| [`pdf/3-flowstate.pdf`](pdf/3-flowstate.pdf) | How much capital can it absorb before it cannot be traded? |

They are designed to be read in that order — each answers a question the previous one raises. All
three carry a section reporting the generator-validation study, which issued the identical frozen
task specification to three frontier models and passed the results through all three benchmarks
unchanged; its pre-registration and full results are in
[`../benchmarks/generator_validation/`](../benchmarks/generator_validation/).

## Everything else in this directory is build input

| Path | What it is |
|---|---|
| `*.tex` | The LaTeX source of each paper |
| `*_numbers.tex` | Generated macro files — every number a paper reports |
| `figures/` | Generated figures |

**No number in a paper is typed by hand.** The `*_numbers.tex` files are written by
`scripts/build_paper_numbers.py`, `scripts/build_flowstate_numbers.py` and
`scripts/build_alphaaudit_numbers.py` from frozen artifacts, and
`scripts/check_paper_numbers.py` fails if a paper uses a macro nothing defines, defines one
nothing uses, or contains a bare numeral in a claim position. Editing a figure in the `.tex` by
hand is therefore not a shortcut — it is a build failure.

## Rebuilding a paper

Regenerate the numbers, then compile twice so cross-references resolve, then place the PDF where
readers expect it:

```
python scripts/build_paper_numbers.py          # regimestress_numbers.tex
python scripts/build_flowstate_numbers.py      # flowstate_numbers.tex
python scripts/build_alphaaudit_numbers.py     # alphaaudit_numbers.tex
python scripts/check_paper_numbers.py          # must PASS before compiling

cd papers
pdflatex -interaction=nonstopmode <slug>.tex
pdflatex -interaction=nonstopmode <slug>.tex
mv <slug>.pdf pdf/<slug>.pdf
```

Auxiliary files (`.aux`, `.log`, `.out`) are gitignored.
