#!/bin/bash

lang=""

# all the files are tsv and ready to put into docx so no need for if clauses
for f in *; do
    echo $f;
    lang=${f%_train_downsampled.tsv};
    # call the python script and to give the language from the filename to it as well as paths for the docx and labels to save to
    cat $f | python3 ../../codedump/translation_pipeline/todocx.py $lang ../preprocessing/ForDeepL/lol/ ../preprocessing/ForDeepL/full_labels/
done
