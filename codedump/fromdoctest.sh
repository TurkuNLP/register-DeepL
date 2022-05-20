#!/bin/bash

dir=""
file="fi_000.docx"
labeldocdame=""

echo $file
dir="./ForDeepL/$file"
echo $dir
labeldocdame=${file%.docx}
#echo $labeldocdame
echo $dir | python3 fromdocx.py $labeldocdame > testi.tsv