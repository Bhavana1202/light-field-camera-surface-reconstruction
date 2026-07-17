# Thesis figures (TikZ)

- `fig1_overview.tex` — high-level architecture pipeline
- `fig2_detail.tex` — module-level architecture with sub-blocks

Both are generic in angular resolution `a` (used for both 7x7 and 8x8).

## Compiling standalone

    \documentclass[tikz, border=10pt]{standalone}
    \usepackage{tikz}
    \usetikzlibrary{arrows.meta, positioning, fit, backgrounds, calc, shapes.geometric}
    \begin{document}
    \input{fig1_overview}
    \end{document}
