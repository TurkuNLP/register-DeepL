#!/bin/bash


sub='gz'

for f in *.gz ; do 
    zcat $f |
    python3 ../change_labeling_for_train.py |
    gzip > ../main_labels_only/${f%.tsv.gz}.modified.tsv.gz
done