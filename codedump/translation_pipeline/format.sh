#!/bin/bash

sub='gz'

for f in * ; do 
if [[ "$f" == *"$sub"* ]];
then
    zcat $f |
    python3 ../../../../codedump/translation_pipeline/format.py | gzip > ../formatted/${f%.tsv.gz}.formatted.tsv.gz
else
    cat $f |
    python3 ../../../../codedump/translation_pipeline/format.py > ../formatted/${f%.tsv}.formatted.tsv
fi;
done
