#!/bin/bash

sub='gz'

for f in fre* ; do 
if [[ "$f" == *"$sub"* ]];
then
    zcat $f |
    python3 ../../codedump/formatSwe_Fre.py |
    python3 ../../codedump/change_labeling_for_train.py > ../formatted/${f%.tsv.gz}.formatted.tsv
else
    cat $f |
    python3 ../../codedump/formatSwe_Fre.py |
    python3 ../../codedump/change_labeling_for_train.py > ../formatted/${f%.tsv}.formatted.tsv
fi;
done
