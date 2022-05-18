import transformers
import datasets
import gzip 
import random
import re
import json

# somehow get the data, cat? or with open
# will there be a problem with the different label annotations?
#  Like will the ai actually learn correctly even though there are different ways of marking the labels
# and will the predictions match the evaluation way of marking the labels


label_names=[] # how do I get a list of all the label names?
# do I need to go through them all and make a list of unique labels? or separate them somehow? 
# I hope there is a ready made list somewhere...
# => next thing is to make a list of the labels included in the picture?


# Each sample has a binary value for each label. 
# THIS IS THE CASE FOR MANY EXAMPLES (the data) BUT IT IS PROBABLY NOT VIABLE HERE SINCE THERE ARE SO MANY OPTIONS?
# apparently one-hot encoding is a thing and it seems to be necessary

# and for the data do I make the labels into a list as well??? => probably yes because of the one-hot encoding lol


file_name = "es_FINAL.tsv.gz" # or whatever the script gives as a parameter

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



random.seed(1234) # remember to shuffle since the data is now in en,fi,swe,fre order !!!!
random.shuffle(data) 

with open("translated-register-data.jsonl", "wt") as f:
    for cols in data:
        item = {
            "text": cols[1],
            "label": label_names.index(cols[0]),    # this I have to change to include many labels (list of them?), one-hot encoding?
        }
        print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)


# unfortunately multilabel examples only show how to do from a ready test and they do not do this at all
# so I probably need some help setting the labels?

file = "translated-register-data.jsonl"
dataset = datasets.load_dataset(
    'json',                             
    data_files={"everything":file, "test":file},    # I need the test set from Veronika? ask Filip and Veronika
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


# then use the XLMR tokenizer (does this work for everyone of these languages?)

model_name = "xlm-roberta-base" # we use the xlmr for tokenizing
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True # use some other method for this?
    )

dataset = dataset.map(tokenize)


# then build model with AutoModelForSequenceClassification?
# XLMRobertaForSequenceClassification
#model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", problem_type="multi_label_classification")
# just set the problem type to multilabel


num_labels = len(label_names)
model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
# a couple examples mentioned that I should have these dictionaries that map labels to integers and back
# but this once again requires me to have a list of all the possible labels
# id2label=id2label,
# label2id=label2id




# then I configure the model

# I probably need to make my own compute_accuracy method based on the sentiment notebook and
# an example notebook for multilabel classification I found online




# and finally train the model

#trainer.train()







#RAMBLING ABOUT THE LABELS
# it seems that english has just the basic labels, sometimes many of them but no sublabels

# swedish and french has base labels first
# and then the corresponding sublabels e.g. NA IP (both base) + OA DS (both sub)

# finnish has sublabel baselabel sublabel baselabel style (?)
# sometimes baselabel sublabel baselabel
# there seem to be some typos for some labels, and for one atleast NA was mentioned twice
# IP was IG instead in a few places
# could look at that more and mark the line numbers
