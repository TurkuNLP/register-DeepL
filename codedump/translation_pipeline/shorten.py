import transformers
import os
import gzip
import sys


def make_shorter(data, length):
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
        if len(tok_outs[i]["offset_mapping"]) <= length+2:
            #had to change -1 to -2 to get it working with smaller sentences than whatever was bigger than the sentence
            # if it was -1 it takes (0,0) which should not happen
            shortened = data[i][1][0:tok_outs[i]["offset_mapping"][:length][-2][1]] 
        else:
            shortened = data[i][1][0:tok_outs[i]["offset_mapping"][:length][-1][1]] 
            
        # print(data[i]) # for comparison to the original text
        print(shortened)
        # print()

        # length comparisons
        # print(len(data[i].split()))
        # print(len(shortened.split()))



data = sys.stdin.readlines() # get data from pipe (cat [FILE])

# split to labels and texts so that we do not shorten the whole line, only the text
# the data has to be formatted to label, text before this because some of original data has other things as well and faulty lines
for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split the labels and text

length = sys.argv[1] # get length of shortened text (number of characters) from a positional argument


#print("file length",len(data))
short_data = make_shorter(data, length)

