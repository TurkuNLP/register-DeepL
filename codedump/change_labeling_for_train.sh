#!/bin/bash


for f in *.tsv ; do  #.gz
    cat $f | #zcat
    python3 ../codedump/change_labeling_for_train.py > ../main_labels_only/original_downsampled/${f%.tsv}_modified.tsv
    # | gzip > ../main_labels_only/${f%.tsv.gz}.modified.tsv.gz
done