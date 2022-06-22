#!/bin/bash

# have to delete every time I run this because I use >> which appends to the file

dir=""
labeldocdame=""

for f in *.docx ; 
do 
    echo $f;
    dir="./$f";
    labeldocdame=${f%.docx};
    # here add whatever the file name is after translation, preferably the language to which it was translated to
    # e.g. /ja.tsv.gz or ${f%_*}_ja.tsv.gz
    echo $dir | python3 ../../../codedump/not_in_current_use/fromdocx.py $labeldocdame | gzip >> ../../../AfterDeepL/FIN_FINAL.tsv.gz
done