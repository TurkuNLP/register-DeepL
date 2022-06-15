#!/bin/bash

sub='gz'

for f in *.tsv ; do 
    if [[ "$f" == *"$sub"* ]];
    then
        echo $f >> label_distribution.txt ; zcat $f | python3 ../../label_distribution.py >> label_distribution.txt
    else
        echo $f >> label_distribution.txt ; cat $f | python3 ../../label_distribution.py >> label_distribution.txt 
    fi;
done


echo "All files" >> label_distribution.txt | cat *.tsv | python3 ../../label_distribution.py >> label_distribution.txt 