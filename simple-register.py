import transformers
import datasets
import gzip 
import random
import re
import json
import sys


from pprint import PrettyPrinter
import logging
pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)

file_name = sys.argv[1]
data=[]
with open(file_name ) as f:
    for line in f:
        #line = line.decode()
        line=line.rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        data.append(cols)

#pprint(data[0])

random.seed(1234) # remember to shuffle since the data is now in en,fi,swe,fre order
random.shuffle(data) 

# get a list of all the unique labels in the data using set which does not allow duplicates

labels = [one[0] for one in data]

labelset = set(labels) #split_labels
unique_labels=list(labelset)

#texts= [one[1] for one in data]
# print(len(unique_labels)) # only 43 when splitting the labels but 358 when not!!!
# print(unique_labels[:10])


with open("translated-register-data.jsonl", "wt") as f:
    for cols in data:
        item = {
            "text": cols[1],
            "label": unique_labels.index(cols[0]), 
        }
        print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)



file = "translated-register-data.jsonl"
dataset = datasets.load_dataset(
    'json',                             
    data_files={"everything":file}, # I need the test set from Veronika? ask Filip and Veronika
    split={
        "train":"everything[:80%]",  
        "validation":"everything[80%:90%]",   
        "test":"everything[90%:]"    
    },
    features=datasets.Features({
        "label":datasets.ClassLabel(names=unique_labels),
        "text":datasets.Value("string")
    })
)        

model_name = "xlm-roberta-large" # we use the xlmr for tokenizing (large?, I used base previously)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True, # do something else other than truncating? the texts now have max 1024 tokens
    )

dataset = dataset.map(tokenize)

model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels))
#model.cuda() # set to use the GPU(?) I've never tried this


trainer_args = transformers.TrainingArguments(
    "../multiclass/checkpoints",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch", #added
    load_best_model_at_end=True,
    num_train_epochs=3, #added
    # eval_steps=100,
    # logging_steps=100,
    learning_rate=0.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    # max_steps=1500
)


data_collator = transformers.DataCollatorWithPadding(tokenizer)

#microF1
from sklearn.metrics import precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro') # switch to micro for multilabel at least, and apparently because it is multiclass binary is not working
  #  acc = accuracy_score(labels, preds)
    return {
   #     'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()

early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5
)


trainer = transformers.Trainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

trainer.train()


eval_results = trainer.evaluate(dataset["test"])

print('F1:', eval_results['eval_f1'])
#print(eval_results)


#RAMBLING ABOUT THE LABELS
# it seems that english has just the basic labels, sometimes many of them but no sublabels

# swedish and french has base labels first
# and then the corresponding sublabels e.g. NA IP (both base) + OA DS (both sub)

# finnish has sublabel baselabel sublabel baselabel style (?)
# sometimes baselabel sublabel baselabel
# there seem to be some typos for some labels, and for one atleast NA was mentioned twice
# IP was IG instead in a few places
# could look at that more and mark the line numbers

# I could probably change the swe and fre to have base+sub, base+sub like the finnish?
# because if the sublabels are separated from the base labels it might just use the sublabel?
# => (unlikely though)
