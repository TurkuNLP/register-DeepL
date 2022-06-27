#!/bin/bash

ALL=0
CURRENT=0
sub='gz'


for f in * ; do 
    if [[ "$f" == *"en_train1"*  ]];
    then
        echo ALL THE EN_TRAIN FILES;
        zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | wc -c
    fi;
    if [[ "$f" == *"$sub"* ]];
    then
        echo $f; zcat $f | wc -c;
        CURRENT=$(zcat $f | wc -c);
        ALL=$(($ALL + $CURRENT ))
    else
        echo $f ; cat $f | wc -c ;
        CURRENT=$(cat $f | wc -c);
        ALL=$(($ALL + $CURRENT ))
    fi;
done
echo ALL OF THE CHARACTERS
echo $ALL

