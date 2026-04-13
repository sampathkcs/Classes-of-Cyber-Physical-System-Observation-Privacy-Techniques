#!/bin/sh
for f in *.pdf
do
    echo "$f"
    pdfcrop "$f" "$f"
    echo ""
done
