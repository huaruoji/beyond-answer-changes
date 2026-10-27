# Paper sources

The current manuscript is **reference-draft.tex**, producing **reference-draft.pdf**.

```sh
cd paper
python3 make_reference_figures.py          # needs matplotlib + numpy
latexmk -pdf -interaction=nonstopmode -halt-on-error reference-draft.tex
python3 verify_reference_draft.py          # needs pypdf for the PDF-layout check
```

`make_reference_figures.py` renders in **Times New Roman**, which is not installed
on every machine. It no longer hard-fails on a missing font: it falls back to the
first available of `Times`, `Liberation Serif`, `Nimbus Roman`, `DejaVu Serif` and
prints a `NOTE:` naming the substitute. A substituted font has different metrics,
so regenerate the shipped figures only where Times New Roman is present. On this
machine it is installed from the Windows font directory, which is the cheapest
route when WSL can see `C:\Windows\Fonts`:

```sh
cp /mnt/c/Windows/Fonts/times{,bd,i,bi}.ttf ~/.local/share/fonts/ && fc-cache -f
```

`--font NAME` forces a family and `--out DIR` renders somewhere other than
`figures/` without clobbering them. The script recomputes the per-class
statistics straight from the prediction JSONL, asserts 33 classes, 1,200 trials
and that the trial-weighted class gap equals `A_diff - A_same` in the frozen
table, and checks that every text box fits the canvas before saving.

The manuscript carries three figures: the full-width change-rate bar chart (with
the change-rate gap and its soundscape-clustered interval as the rightmost group),
the single-column class-level accuracy-versus-gap scatter, and the single-column
empirical cumulative distribution of class-level gaps.

- `chapters/`: current manuscript text. `06-limitations.tex` sits between the
  results and the conclusion. The Acknowledgments section in `reference-draft.tex`
  contains the funding and conflict-of-interest statements and stays within the
  first four pages because the paper kit limits non-reference content on page five.
- `reference-refs.bib`: current references, formatted with the official `IEEEbib.bst`.
- `figures/reference_*.pdf`: current figures; generated from the frozen evidence table.
  `reference_identity` carries the change-rate gap with its soundscape-clustered interval.
- `official-template/`: unmodified files downloaded from the ICASSP 2027 official template ZIP. Its sample PDF explains the formatting; it is not our paper.
- `archive/template-draft/`: historical manuscript, bibliography, figures, generator, and build files, preserved together. Compile its `Template.tex` from that directory. Its metric interpretation and some figure labels are superseded; use it as a visual reference, not a source of current conclusions. The archived generator retains its original paths for provenance and should not be run in place.

Official ZIP: https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/ICASSP2027_Paper_Templates.zip

The active `spconf.sty` and `IEEEbib.bst` are byte-identical to the official download. The manuscript uses the template's default `hyperref` settings, numeric citations and `IEEEbib` bibliography. No margin, line-spacing, citation-style or font-size override is applied. `amssymb` supplies mathematical symbols only.
