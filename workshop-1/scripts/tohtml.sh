list=$(ls ../[0-9]*.md)
echo $list

pandoc $list -o slides.html -s --css custom.css


