#!/bin/bash
set -euo pipefail
mkdir -p converted_pngs
DPI="${DPI:-300}"   # change to 600 for ultra crisp

for file in *.svg; do
  base="${file%.svg}"
  inkscape "$file" \
    --export-type=png \
    --export-dpi="$DPI" \
    --export-area-drawing \
    --export-filename="converted_pngs/${base}.png"
done
