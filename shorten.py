#!/usr/bin/env python3

import transformers # pip install [package] --user (if more packages needed)
import os
import gzip
import sys

# module load pytorch before starting again on csc



# a margninal change between 1536 and 1024, line counts matter more



def make_shorter(data):
    model_name = "xlm-roberta-base" # we use the xlmr for tokenizing
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example,
            return_offsets_mapping=True, # give where the tokens start and end
            return_special_tokens_mask=False #tell so that there are no seps etc.
        )
    
    # this was just for testing it works

    # text ="Tämä on testi, joka toivottavasti toimii." 
    # print(text)
    # tok_out = tokenize(text)
    # print(tok_out)
    # if len(tok_out["offset_mapping"]) <= 1536+2:
    #     text2 = text[0:tok_out["offset_mapping"][:1536][-2][1]] 
    # else:
    #     text2 = text[0:tok_out["offset_mapping"][:1536][-1][1]] 
    # print(text2)
    # print(len(text2.split()))



    # # this tokenizes the whole thing
    tok_outs =list(map(tokenize, data))
    
    for i in range(len(data)):
        #print(tok_outs[i]["offset_mapping"])
        if len(tok_outs[i]["offset_mapping"]) <= 1536+2:
            #had to change -1 to -2 to get it working with smaller sentences than whatever was bigger than the sentence
            # if it was -1 it takes (0,0) which should not happen
            shortened = data[i][0:tok_outs[i]["offset_mapping"][:1024][-2][1]] 
        else:
            shortened = data[i][0:tok_outs[i]["offset_mapping"][:1024][-1][1]] 
        # print(data[i])

        
        print(shortened)
        # print()

        # print(len(data[i].split()))
        # print(len(shortened.split()))
        # instead of this, on the command line do wc -w, or wc -c


#This commented thing is unnenecessary right now

# gzname="./multilingual-register-data/CORE/en_train2.tsv.gz"
# with gzip.open (gzname, 'rb') as f:
#    data = f.readlines().decode()

# fname= "./multilingual-register-data/FinCORE/v2_april21/fi_dev.tsv" #this file is just for testing
# with open (fname) as f:
#     data = f.readlines()


# zcat or cat [file name] | python3 shorten.py | the python script prints and is then put to a file?
# zcat multilingual-register-data/CORE/en_train1.tsv.gz | python3 shorten.py | gzip > en_train1.truncated.tsv.gz

    # do this in the multlingual-register-data-new directory
# for f in *.tsv ; do cat $f | python3 ../shorten.py > ../preprocessed_texts/${f%.tsv}.truncated.tsv ; done
# for f in *.gz ; do zcat $f | python3 ../shorten.py | gzip > ../preprocessed_texts/${f%.tsv.gz}.truncated.tsv.gz ; done



data = sys.stdin.readlines()


# to make new files for each language
#  and then later count the words in the files
# also keep train and dev and test separate


#print("file length",len(data))
short_data = make_shorter(data)

