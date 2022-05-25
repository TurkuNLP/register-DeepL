import transformers
import datasets
import gzip 
import random
import re
import json
import torch
import sys


from pprint import PrettyPrinter
import logging
pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)

# file_names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
original = sys.stdin.readlines()
dev_set= sys.argv[1]
test_set = sys.argv[2]

data=[]
for line in original:
    line=line.rstrip("\n")
    cols=line.split("\t")
    if len(cols)!=2: #skip weird lines that don't have the right number of columns
        continue
    data.append(cols)

tests=[]
with gzip.open(test_set, 'rb') as f:
    for line in f:
        line = line.decode()
        line=line.rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        tests.append(cols)

dev=[]
with gzip.open(dev_set, 'rb') as f:
    for line in f:
        line = line.decode()
        line=line.rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        dev.append(cols)

random.seed(1234) # remember to shuffle since the data is now in en,fi,swe,fre order
random.shuffle(data) 

# get a list of all the unique labels in the data using set which does not allow duplicates
def all_possible_labels(data):
    labels = [one[0] for one in data]
    #this splits all of the labels into their own thing for multilabeling
    split_labels= []
    for labeled in labels:
        labeleds = labeled.split()
        for label in labeleds:
            split_labels.append(label)

    return split_labels

unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]
# labels1 = all_possible_labels(data)
# labels2 = all_possible_labels(tests)
# labels3 =  all_possible_labels(dev)
# all_labels= labels1 + labels2 + labels3

# labelset = set(all_labels) #split_labels
# unique_labels=list(labelset)
# print(len(unique_labels))

def split_labels(data):
    texts= [one[1] for one in data]

    # split the labels into a list in the data
    for i in range(len(data)):
        labeledlist = data[i][0]
        lablist = labeledlist.split()
        data[i][0] = lablist

    labels = [one[0] for one in data]
    return texts, labels

texts, labels = split_labels(data)
test_texts, test_labels = split_labels(tests)
dev_texts, dev_labels = split_labels(dev)

# one-hot encoding! THIS ONLY ENCODES THE LABELS THAT ARE PRESENT IN THE DATA (SOME MAY NOT EXIST)
import pandas as pd

# first for data
df = pd.DataFrame({
    "text": texts,
    "labels": labels
})

#then for test data
dftest = pd.DataFrame({
    "text": test_texts,
    "labels": test_labels
})

dfdev = dftest = pd.DataFrame({
    "text": dev_texts,
    "labels": dev_labels
})
# put on top of each other
vertical_stack = pd.concat([df, dfdev, dftest], axis=0)
vertical_stack.reset_index(drop=True)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(vertical_stack.labels)
# print(yt[0])
# print(mlb.inverse_transform(yt[0].reshape(1,-1)))
# print(mlb.classes_)

ytlist= []
for labs in yt:
  ytlist.append(labs)

length = len(df.index)
length2 = len(dfdev.index)
length3 = len(dftest.index)

# redo the dataframes with the encoded labels
df = pd.DataFrame({
    "text": vertical_stack.iloc[:length].text,
    "labels": ytlist[:length]
})

dfdev = pd.DataFrame({
    "text": vertical_stack.iloc[length:length+length2-2].text,
    "labels": ytlist[length:length+length2-2]
})

dftest = pd.DataFrame({
    "text": vertical_stack.iloc[length+length2:length+length2+length3-3].text,
    "labels": ytlist[length+length2:length+length2+length3-3]
})



# make train and test into a dataset
train = datasets.Dataset.from_pandas(df)
test = datasets.Dataset.from_pandas(dftest)
dev = datasets.Dataset.from_pandas(dfdev)
dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})

# with open("register-data.jsonl", "wt") as f:
#     for cols in data:
#         item = {
#             "text": cols[1],
#             "label": unique_labels.index(cols[0]), 
#         }
#         print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)
        
# with open("register-tests.jsonl", "wt") as f:
#     for cols in tests:
#         item = {
#             "text": cols[1],
#             "label": unique_labels.index(cols[0]), 
#         }
#         print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)

# with open("register-dev.jsonl", "wt") as f:
#     for cols in dev:
#         item = {
#             "text": cols[1],
#             "label": unique_labels.index(cols[0]), 
#         }
#         print(json.dumps(item,ensure_ascii=False,sort_keys=True),file=f)




# file = "register-data.jsonl"
# file2= "register-tests.jsonl"
# file3= "register-dev.jsonl"
# dataset = datasets.load_dataset(
#     'json',                             
#     data_files={"train":file, "dev": file2, "test": file3},
#     split={
#         "train":"train",  
#         "validation":"dev",   
#         "test":"test"    
#     },
#     features=datasets.Features({
#         "label":datasets.ClassLabel(names=unique_labels),
#         "text":datasets.Value("string")
#     })
# )        

model_name = "xlm-roberta-base" # we use the xlmr for tokenizing (large?, I used base previously)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True, # do something else other than truncating? the texts now have max 1024 tokens
    )

dataset = dataset.map(tokenize)

model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels), problem_type="multi_label_classification")
if torch.cuda.is_available():
    model = model.cuda()


# this I probably have to change by a lot since there is way more data in the original files
trainer_args = transformers.TrainingArguments(
    "../benchmark/checkpoints",
    evaluation_strategy="epoch",
    logging_strategy="epoch",   # changed both from steps
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    #eval_steps=100,
    #logging_steps=100,
    learning_rate=0.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    #max_steps=1500
)


data_collator = transformers.DataCollatorWithPadding(tokenizer)

# #microF1
# from sklearn.metrics import precision_recall_fscore_support
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro') # switch to micro for multilabel at least, and apparently because it is multiclass binary is not working
#   #  acc = accuracy_score(labels, preds)
#     return {
#    #     'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

#compute accuracy and loss
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.3):  # with 0.1 the spanish data is not doing great
    # the treshold has to be really low because the probabilities of the predictions are not great, could even do without any treshold then? or find one that works best between 0.1 and 0.5 (with english)
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels) # why is the sigmoid applies? could do without it
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

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

# we do this because for some reason the problem type does not change the loss function in the trainer
class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

# and finally train the model
trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)
trainer.train()


eval_results = trainer.evaluate(dataset["test"])

print('F1:', eval_results['eval_f1'])
#print(eval_results)