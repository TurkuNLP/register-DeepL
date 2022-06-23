#!/bin/bash


for f in *.tsv.gz ; do  #.gz
    zcat $f | #zcat
    python3 ../codedump/change_labeling_for_train.py | gzip > full_labels/${f%.tsv.gz}_full.tsv.gz  #| gzip > ../main_labels_only/${f%.tsv.gz}.modified.tsv.gz
    #> ../main_labels_only/${f%.tsv}_modified.tsv \
    
done