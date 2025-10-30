files=$(ls ../[0-9]*.md)
#files="../30-numba.md ../70-compare.md"

HEADER="$(dirname "$0")/pandoc-latex-header.tex"

pandoc -s -o ../ws3-notes.pdf $files \
  --pdf-engine=xelatex \
  --include-in-header="$HEADER" \
  --toc --toc-depth=2 \
  --number-sections \
  -V geometry:margin=0.75in \
  -V fontsize=12pt
