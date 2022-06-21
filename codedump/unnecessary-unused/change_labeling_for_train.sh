#!/bin/bash


for f in fre* ; do  #.gz
    cat $f | #zcat
    python3 ../../../codedump/unnecessary-unused/change_labeling_for_train.py > ../main_labels_only/${f%.tsv}_modified.tsv  #| gzip > ../main_labels_only/${f%.tsv.gz}.modified.tsv.gz
    #> ../main_labels_only/${f%.tsv}_modified.tsv \
    
done