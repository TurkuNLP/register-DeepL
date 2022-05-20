#!/bin/bash

# do this in the multilingual-register-data-new folder

sub='gz'

for f in * ; do 
if [[ "$f" == *"$sub"* ]];
then
    zcat $f |
    python3 ../shorten.py |
    gzip > ../preprocessed_texts/${f%.tsv.gz}.truncated.tsv.gz
else
    cat $f |
    python3 ../shorten.py > ../preprocessed_texts/${f%.tsv}.truncated.tsv 
fi;
done