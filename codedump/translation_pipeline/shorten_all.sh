#!/bin/bash

sub='gz'

for f in * ; do 
if [[ "$f" == *"$sub"* ]];
then
    zcat $f |
    python3 ../../../../codedump/translation_pipeline/shorten.py |
    gzip > ../../../preprocessing/preprocessed_texts/${f%.tsv.gz}.truncated.tsv.gz
else
    cat $f |
    python3 ../../../../codedump/translation_pipeline/shorten.py > ../../../preprocessing/preprocessed_texts/${f%.tsv}.truncated.tsv 
fi;
done