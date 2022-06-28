#!/bin/bash

sub='gz'

for f in * ; do 
    if [[ "$f" == *"txt"* ]];
    then
        continue
    fi;
    if [[ "$f" == *"$sub"* ]];
    then
        echo $f >> label_distribution.txt ; zcat $f | python3 ../../codedump/label_stuff/label_distribution.py >> label_distribution.txt
    else
        echo $f >> label_distribution.txt ; cat $f | python3 ../../codedump/label_stuff/label_distribution.py >> label_distribution.txt 
    fi;
done


echo "All files" >> label_distribution.txt | cat *.tsv | python3 ../../codedump/label_stuff/label_distribution.py >> label_distribution.txt 