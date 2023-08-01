#!/usr/bin/env bash
make clean
rm -r source/reference
sphinx-apidoc -o source/reference ../alea
make html #SPHINXOPTS="-W --keep-going -n"
