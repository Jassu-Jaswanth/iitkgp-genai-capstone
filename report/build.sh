#!/usr/bin/env sh
# Build report.pdf from main.tex.
#   ./build.sh          uses latexmk if TeX Live is installed, otherwise Docker (no TeX install needed)
#   ./build.sh clean    remove build files
set -e
cd "$(dirname "$0")"
IMAGE="${TEXLIVE_IMAGE:-texlive/texlive:latest-medium}"
if [ "$1" = "clean" ]; then rm -rf build; exit 0; fi
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=report main.tex
else
  echo "latexmk not found; compiling in Docker ($IMAGE, first run downloads it)"
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work "$IMAGE" \
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=report main.tex
fi
cp build/report.pdf report.pdf
pages=$( (command -v pdfinfo >/dev/null && pdfinfo report.pdf | awk '/^Pages/{print $2}') || echo "?")
echo "report.pdf: $pages pages (limit 12)"
