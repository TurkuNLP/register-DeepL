#!/bin/bash


for f in *.tsv.gz ; do  #.gz
    zcat $f | #cat
    python3 ../codedump/unnecessary-unused/change_labeling_for_train.py | gzip > main_labels_only/${f%.tsv.gz}.modified.tsv.gz
    #> main_labels_only/${f%.tsv}_modified.tsv \
    
done