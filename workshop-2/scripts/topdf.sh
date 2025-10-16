
files=$(ls ../[0-9]*.md)

HEADER="$(dirname "$0")/pandoc-latex-header.tex"

pandoc -s -o ../ws2-notes.pdf $files \
  --pdf-engine=xelatex \
  --include-in-header="$HEADER" \
  --toc --toc-depth=2 \
  -V geometry:margin=0.75in \
  -V fontsize=12pt

