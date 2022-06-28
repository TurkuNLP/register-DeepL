#!/bin/bash


for f in *.tsv.gz ; do  #.gz
    zcat $f | #zcat
    # set arg to full if want to change to full labels and write something else to get only upper labels
    # HOX! positional arguments are necessary so something must be written there
    python3 ../codedump/label_stuff/change_labeling.py full | gzip > full_labels/${f%.tsv.gz}_full.tsv.gz  #| gzip > ../main_labels_only/${f%.tsv.gz}.modified.tsv.gz
    #> ../main_labels_only/${f%.tsv}_modified.tsv \
    
done