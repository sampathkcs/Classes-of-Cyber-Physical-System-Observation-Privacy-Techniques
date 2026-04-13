#!/bin/sh
for f in *.pdf
do
    echo "$f"
    pdffonts "$f"
    echo ""
done
