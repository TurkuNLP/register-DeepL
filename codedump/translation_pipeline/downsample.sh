#!/bin/bash


# File paths
PYTHON="../../../codedump/translation_pipeline/downsample.py"
TARGET="../../downsampled/${f%.truncated.tsv}.downsampled.tsv"


# with these settings and the fifth biggest label char amount being the cap we get just under 25 million
sub='gz'
ALL=0
CURRENT=0
ENG=76.65 #77.7 #76.7 #75.5 # 76.5 with other labels -> 1% difference
# with 40% decrease there is still almost 23 million characters
# with 60% 15 million
# 77 seems to be perfect with a bit less than 9 million (1 million char cap + only id)
FIN=56.2   #53
# with 40% decrease there is still almost 10 million
# with 53% there is still 7 million which is on par with the downsampled en (1 million char cap + only id)

# swe and fre then have about 5 million each
# => total is a bit less than 25 million
SWE=8.5

# if there is one argument given, it is just the percent, if there is two, there is label cap and percent
# and if there is no argument it just takes out faulty lines if there are any
for f in *; do
    if [[ "$f" == *"en_train1"* ]];
    then
        echo ALL THE EN_TRAIN FILES;
        # zcat every train file because they continue from each other
        zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | python3 $PYTHON 4 $ENG  > ../../downsampled/${f%1.truncated.tsv.gz}.downsampled.tsv
        CURRENT=$(zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | python3 $PYTHON 4 $ENG | wc -c);
        echo $CURRENT;
        ALL=$(($ALL + $CURRENT ))
        continue
    fi;
    if [[ "$f" == *"fi_train"* ]];
    then 
        echo $f;
        zcat fi_train.truncated.tsv.gz | python3 $PYTHON 4 $FIN > ../../downsampled/${f%.truncated.tsv.gz}.downsampled.tsv
        CURRENT=$(zcat fi_train.truncated.tsv.gz | python3 $PYTHON 4 $FIN | wc -c);
        echo $CURRENT;
        ALL=$(($ALL + $CURRENT ))
        continue
    fi;
    # # the rest of the fi files
    # if [[ "$f" == *"fi"* ]];
    # then
    #     echo $f;
    #     cat $f | python3 $PYTHON 4 $FIN > $TARGET
    #     CURRENT=$(cat $f | python3 $PYTHON 4 $FIN | wc -c);
    #     ALL=$(($ALL + $CURRENT ))
    #     continue
    # fi;
    # # skip the rest of the eng training files
    # if [[ "$f" == *"en_train"* ]];
    # then
    #     continue
    # fi;
    if [[ "$f" == *"$sub"* ]];
    then
        # # downsample english dev and test as well
        # echo $f;
        # zcat $f | python3 $PYTHON 4 $ENG > ../../downsampled/${f%.truncated.tsv.gz}.downsampled.tsv
        # CURRENT=$(zcat $f | python3 $PYTHON 4 $ENG | wc -c);
        # ALL=$(($ALL + $CURRENT ))
        continue
    else
        # downsample swe train set
        if [[ "$f" == *"swe_train"* ]];
        then
            echo $f ; cat $f | python3 $PYTHON $SWE > $TARGET
            CURRENT=$(cat $f | python3 $PYTHON $SWE | wc -c);
            echo $CURRENT;
            ALL=$(($ALL + $CURRENT ))
            continue
        fi;

        # for files that do not need explicit downsampling
        if [[ "$f" == *"train"* ]];
        then
            echo $f ; cat $f | python3 $PYTHON > $TARGET
            CURRENT=$(cat $f | python3 $PYTHON | wc -c);
            echo $CURRENT;
            ALL=$(($ALL + $CURRENT ))
        fi;

        # # if the file in question is french or swedish (dev + test)
        # if [[ "$f" == *"fre"* || "$f" == *"swe"* ]]
        # then
        #     echo $f;
        #     cat $f | python3 $PYTHON 20 > $TARGET
        #     CURRENT=$(cat $f | python3 $PYTHON 20 | wc -c);
        #     ALL=$(($ALL + $CURRENT ))
        # fi;
    fi;
done
echo ALL OF THE CHARACTERS
echo $ALL