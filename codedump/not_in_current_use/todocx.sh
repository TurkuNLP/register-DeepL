#!/bin/bash

lang=""

# all the files are tsv and ready to put into docx so no need for if clauses
for f in *full.tsv; do
    echo $f;
    lang=${f%_train_downsampled_full.tsv};
    # call the python script and try to give the language from the filename to it
    cat $f | python3 ../codedump/not_in_current_use/todocx.py $lang
done
