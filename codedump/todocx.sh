#!/bin/bash

lang=""

# all the files are tsv and ready to put into docx so no need for if clauses
for f in *; do
    echo $f;
    lang=${f%_train.downsampled.tsv};
    # call the python script and try to give the language from the filename to it
    cat $f | python3 ../todocx.py $lang
done
