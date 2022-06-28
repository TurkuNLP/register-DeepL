#!/bin/bash

sub='gz'

for f in * ; do 
    if [[ "$f" == *"en_train1"* ]];
    then
        echo ALL THE EN_TRAIN FILES;
        # zcat every train file because they continue from each other
        zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | python3 ../../../codedump/translation_pipeline/count_chars_labelled.py
    fi
    if [[ "$f" == *"en_train"* ]]
    then
        continue
    fi
    if [[ "$f" == *"$sub"* ]];
    then
        echo $f; zcat $f | python3 ../../../codedump/translation_pipeline/count_chars_labelled.py
    else
        echo $f ; cat $f | python3 ../../../codedump/translation_pipeline/count_chars_labelled.py
    fi;
done