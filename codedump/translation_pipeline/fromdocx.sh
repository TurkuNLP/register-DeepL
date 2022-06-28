#!/bin/bash

# have to delete every time I run this because I use >> which appends to the file

dir=""
labeldocname=""

for f in *.docx ; 
do 
    echo $f;
    dir="./$f";
    # this only works because deepL adds a space after the original name e.g. en_000.docx => en_000 zh.docx
    # the script then takes the argument but the argument set here is actually split into two arguments
    labeldocname=${f%.docx};
    # here add whatever the file name is after translation, preferably the language to which it was translated to
    # e.g. /ja.tsv.gz or ${f%_*}_ja.tsv.gz
    echo $dir | python3 ../../../codedump/translation_pipeline/fromdocx.py $labeldocname | gzip >> ../../AfterDeepL/new_FINAL.tsv.gz
done