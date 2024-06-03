#!/bin/bash

pdflatex graphs.tex
#pdf270 graphs.pdf
pdfjam --landscape --angle 270 graphs.pdf
mv graphs-*.pdf graphs.pdf

