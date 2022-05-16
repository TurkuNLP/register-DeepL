#!/bin/bash

dir=""
labeldocdame=""

for f in *.docx ; 
do 
    echo $f;
    dir="./$f";
    labeldocdame=${f%.docx};
    # here add whatever the file name is after translation, preferably the language to which it was translated to
    # e.g. /ja.tsv.gz or ${f%_*}_ja.tsv.gz
    echo $dir | python3 ../fromdocx.py $labeldocdame | gzip >> ../AfterDeepL/${f%_*}_FINAL.tsv.gz
done