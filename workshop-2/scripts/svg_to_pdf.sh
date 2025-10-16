#!/usr/bin/env bash
set -euo pipefail

# Convert SVG files in workshop-2/assets to PDF fallbacks suitable for LaTeX
# Usage:
#  ./svg_to_pdf.sh              # converts all .svg in ../assets
#  ./svg_to_pdf.sh file.svg     # converts single file

ASSETS_DIR="$(dirname "$0")/../assets"

if [ "$#" -eq 0 ]; then
  svgs=("$ASSETS_DIR"/*.svg)
else
  svgs=("$@")
fi

# Detect available converter
converter=""
if command -v rsvg-convert >/dev/null 2>&1; then
  converter=rsvg-convert
elif command -v inkscape >/dev/null 2>&1; then
  converter=inkscape
elif python - <<'PY' 2>/dev/null
import importlib,sys
if importlib.util.find_spec('cairosvg'):
    sys.exit(0)
sys.exit(1)
PY
then
  converter=cairosvg
else
  echo "No SVG->PDF converter found. Install librsvg2-bin, inkscape, or cairosvg." >&2
  exit 2
fi

for svg in "${svgs[@]}"; do
  [ -e "$svg" ] || continue
  base="${svg%.svg}"
  pdf="${base}.pdf"
  echo "Converting $svg -> $pdf using $converter"
  case "$converter" in
    rsvg-convert)
      rsvg-convert -f pdf -o "$pdf" "$svg"
      ;;
    inkscape)
      # Inkscape >=1.0
      inkscape "$svg" --export-type=pdf --export-filename="$pdf"
      ;;
    cairosvg)
      python -c "import cairosvg,sys;cairosvg.svg2pdf(url=sys.argv[1], write_to=sys.argv[2])" "$svg" "$pdf"
      ;;
  esac
done

echo "Done."
