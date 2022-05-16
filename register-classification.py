import transformers
import datasets
import gzip 
import random

# somehow get the data, cat? or with open

file_name = "es_FINAL.tsv.gz"

import re
import json

label_names=[] # how do I get a list of all the label names?
data=[]
with gzip.open(file_name, 'rb') as f:
    for line in f:
        line=line.decode.rstrip("\n")
        if not line or line.startswith("#"): #skip empty and comments (incl. header)
            continue
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        data.append(cols)



random.seed(1234) # is this necessary since the data is already shuffled before doing the downsampling?
random.shuffle(data) 

with open("translated-register-data.jsonl", "wt") as f:
    for cols in data:
        item = {
            "text": cols[1],
            "label": label_names.index(cols[2]),    # translate from label strings to integers
        }
        print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)


file = "translated-register-data.jsonl"
dataset = datasets.load_dataset(
    'json',                             
    data_files={"everything":file, "test":file},    # I need the test set from Veronika?
    split={
        "train":"everything[:80%]",  
        "validation":"everything[80%:90%]",   
        "test":"test"    
    },
    features=datasets.Features({
        "label":datasets.ClassLabel(names=label_names), # here I need to set the label_names list but I do not have it yet.....
        "text":datasets.Value("string")
    })
)

# then use the XLMR tokenizer?

model_name = "xlm-roberta-base" # we use the xlmr for tokenizing
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True,
    )

dataset = dataset.map(tokenize)




# then build model with AutoModelForSequenceClassification