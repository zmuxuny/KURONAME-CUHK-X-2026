#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >build/pass1.txt
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >build/pass2.txt
cp build/main.pdf KURONAME_Technical_Report.pdf
