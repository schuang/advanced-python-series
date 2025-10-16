list=$(ls ../[0-9]*.md)
echo $list

pandoc $list --pdf-engine=lualatex  -o ../ws1-notes.pdf -V geometry:margin=0.5in -V fontsize=12pt --toc --toc-depth=2


