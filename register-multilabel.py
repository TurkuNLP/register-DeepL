import transformers
import datasets
import gzip 
import random
import re
import json
import sys
import torch

# something needs to be done about the test sets different labels


file_name = sys.argv[1]
test_name = sys.argv[2]

data=[]
with gzip.open(file_name, 'rb') as f:
    for line in f:
        line=line.decode().rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        data.append(cols)

tests=[]
with open(test_name) as f:
    for line in f:
        line=line.rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        tests.append(cols)

def all_possible_labels(data):
    labels = [one[0] for one in data]
    #this splits all of the labels into their own thing for multilabeling
    split_labels= []
    for labeled in labels:
        labeleds = labeled.split()
        for label in labeleds:
            split_labels.append(label)

    return split_labels

# manual list of the main labels
unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]
# # get a list of all the unique labels in the data + test using set which does not allow duplicates
# labels1 = all_possible_labels(data)
# labels2 = all_possible_labels(tests)
# split_labels = labels1 + labels2
# labelset = set(split_labels) #split_labels
# unique_labels=list(labelset)


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


random.seed(1234) # remember to shuffle since the data is now in en,fi,swe,fre order !!!!
random.shuffle(data) 


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
# put on top of each other
vertical_stack = pd.concat([df, dftest], axis=0)
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
length2 = len(dftest.index)

# redo the dataframes with the encoded labels
df = pd.DataFrame({
    "text": vertical_stack.iloc[:length].text,
    "labels": ytlist[:length]
})

dftest = pd.DataFrame({
    "text": vertical_stack.iloc[length2:].text,
    "labels": ytlist[length2:]
})


# make train and test into a dataset
train = datasets.Dataset.from_pandas(df)
test = datasets.Dataset.from_pandas(dftest)
train, dev = train.train_test_split(test_size=0.2).values()
dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})


# then use the XLMR tokenizer

model_name = "xlm-roberta-large"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True # use some other method for this?
    )

dataset = dataset.map(tokenize) # NOW THERE IS A PROBLEM HERE!! 
#TypeError: Values in `DatasetDict` should of type `Dataset` but got type '<class 'str'>'

#label ids to floats so that the trainer accepts these!
dataset.set_format(type="torch", columns=['labels', 'input_ids', 'attention_mask'])
dataset = dataset.map(lambda x : {"float_labels": x["labels"].to(torch.float32)}, remove_columns=["labels", "text"])
dataset = dataset.rename_column("float_labels", "labels")


num_labels = len(unique_labels)
model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
# a couple examples mentioned that I should have these dictionaries that map labels to integers and back
# id2label=id2label,
# label2id=label2id


trainer_args = transformers.TrainingArguments(
    "../multilabel/checkpoints", # change this to put the checkpoints somewhere else
    evaluation_strategy="epoch",
    logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    learning_rate=0.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)

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