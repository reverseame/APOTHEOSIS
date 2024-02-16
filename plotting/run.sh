#!/bin/bash

mkdir graphs

for file in `ls *.out`
do
    filename="${file%.*}"
    python3 plot.py $file
    pdflatex graphs.tex
    mv graphs.pdf graphs/$filename".pdf"
done
cp plots2.tex plots.tex
pdflatex graphs.tex
rm graphs-*.pdf
pdfjam graphs.pdf '2,4,6,8'
pdf270 graphs-*.pdf
mv graphs-*270.pdf graphs/.

