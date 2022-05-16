#!/bin/bash

# with dev and test sets taken into account we will have to downsample even swedish and french
# right now nothing is really done to the swe and fre, just faulty lines are taken out


# with these settings and the fifth biggest label char amount being the cap we get just under 25 million
sub='gz'
ALL=0
CURRENT=0
ENG=75.5 # 76.5 with other labels
# with 40% decrease there is still almost 23 million characters
# with 60% 15 million
# 77 seems to be perfect with a bit less than 9 million
FIN=53
# with 40% decrease there is still almost 10 million
# with 53% there is still 7 million which is on par with the downsampled en

# swe and fre then have about 5 million each
# => total is a bit less than 25 million


# if there is one argument given, it is just the percent, if there is two, there is label cap and percent
# and if there is no argument it just takes out faulty lines if there are any
for f in *; do
    if [[ "$f" == *"en_train1"* ]];
    then
        echo ALL THE EN_TRAIN FILES;
        # zcat every train file because they continue from each other
        zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | python3 ../downsample.py 4 $ENG  > ../downsampled/${f%1.truncated.tsv.gz}.downsampled.tsv # | wc -c;
        CURRENT=$(zcat en_train1.truncated.tsv.gz en_train2.truncated.tsv.gz en_train3.truncated.tsv.gz en_train4.truncated.tsv.gz | python3 ../downsample.py 4 $ENG | wc -c);
        echo $CURRENT;
        ALL=$(($ALL + $CURRENT ))
        continue
    fi;
    if [[ "$f" == *"fi_train"* ]];
    then 
        echo $f;
        zcat fi_train.truncated.tsv.gz | python3 ../downsample.py 4 $FIN > ../downsampled/${f%.truncated.tsv.gz}.downsampled.tsv #| wc -c;
        CURRENT=$(zcat fi_train.truncated.tsv.gz | python3 ../downsample.py 4 $FIN | wc -c);
        echo $CURRENT;
        ALL=$(($ALL + $CURRENT ))
        continue
    fi;
    # # the rest of the fi files
    # if [[ "$f" == *"fi"* ]];
    # then
    #     echo $f;
    #     cat $f | python3 ../downsample.py 4 $FIN | wc -c;
    #     CURRENT=$(cat $f | python3 ../downsample.py 4 $FIN | wc -c);
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
        # zcat $f | python3 ../downsample.py 4 $ENG | wc -c;
        # CURRENT=$(zcat $f | python3 ../downsample.py 4 $ENG | wc -c);
        # ALL=$(($ALL + $CURRENT ))
        continue
    else
        if [[ "$f" == *"train"* ]];
        then
            echo $f ; cat $f | python3 ../downsample.py > ../downsampled/${f%.truncated.tsv}.downsampled.tsv # | wc -c;
            CURRENT=$(cat $f | python3 ../downsample.py | wc -c);
            echo $CURRENT;
            ALL=$(($ALL + $CURRENT ))
        fi;

        # # if the file in question is french or swedish (dev + test)
        # if [[ "$f" == *"fre"* || "$f" == *"swe"* ]]
        # then
        #     echo $f;
        #     cat $f | python3 ../downsample.py 20 | wc -c;
        #     CURRENT=$(cat $f | python3 ../downsample.py 20 | wc -c);
        #     ALL=$(($ALL + $CURRENT ))
        # fi;
    fi;
done
echo ALL OF THE CHARACTERS
echo $ALL