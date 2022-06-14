#!/bin/bash

sub='gz'

for f in en* ; do 
if [[ "$f" == *"$sub"* ]];
then
    zcat $f |
    python3 ../../codedump/formatEn_FIN.py | gzip > ../formatted/${f%.tsv.gz}.formatted.tsv.gz
else
    cat $f |
    python3 ../../codedump/formatEn_FIN.py > ../formatted/${f%.tsv}.formatted.tsv
fi;
done
