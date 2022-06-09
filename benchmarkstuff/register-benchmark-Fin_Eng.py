import transformers
import datasets
import gzip 
import sys
import torch
import argparse
from pprint import PrettyPrinter
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)

def arguments():
    #parser for the optional arguments related to hyperparameters
    parser = argparse.ArgumentParser(
        description="A script for getting register labeling benchmarks",
        epilog="Made by Anni Eskelinen"
    )
    parser.add_argument('train_set')
    parser.add_argument('dev_set')
    parser.add_argument('test_set')
    parser.add_argument('--treshold', type=float, default=0.5,
        help="The treshold which to use for predictions, used in evaluation"
    )
    parser.add_argument('--batch', type=int, default=8,
        help="The batch size for the model"
    )
    parser.add_argument('--epochs', type=int, default=3,
        help="The number of epochs to train for"
    )
    parser.add_argument('--learning', type=float, default=8e-6,
        help="The learning rate for the model"
    )
    args = parser.parse_args()

    def openFiles(data):
        lines=[]
        if "gz" in data:
            with gzip.open(data, 'rb') as f:
                for line in f:
                    line = line.decode()
                    line=line.rstrip("\n")
                    cols=line.split("\t")
                    if len(cols)!=2: #skip weird lines that don't have the right number of columns
                        continue
                    lines.append(cols)
        else:
            with open(data) as f:
                for line in f:
                    line=line.rstrip("\n")
                    cols=line.split("\t")
                    if len(cols)!=2: #skip weird lines that don't have the right number of columns
                        continue
                    lines.append(cols)
        return lines

    # read train files from the pipeline
    original = sys.stdin.readlines()
    data=[]
    for line in original:
        line=line.rstrip("\n")
        cols=line.split("\t")
        if len(cols)!=2: #skip weird lines that don't have the right number of columns
            continue
        data.append(cols)
        
    tests=openFiles(args.test_set)
    dev=openFiles(args.dev_set)

    return args, data, dev, tests

args, train, dev, tests = arguments() 

# random.seed(1234) # remember to shuffle
# random.shuffle(data) 

unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]

def split_labels(data):
    texts= [one[1] for one in data]

    # split the labels into a list in the data
    for i in range(len(data)):
        labeledlist = data[i][0]
        lablist = labeledlist.split()
        data[i][0] = lablist

    labels = [one[0] for one in data]
    return texts, labels

def buildDatasets(data, dev, tests):
    texts, labels = split_labels(data)
    test_texts, test_labels = split_labels(tests)
    dev_texts, dev_labels = split_labels(dev)
    # one-hot encoding! THIS ONLY ENCODES THE LABELS THAT ARE PRESENT IN THE DATA (SOME MAY NOT EXIST THERE)

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

    # make train, dev and test into a dataset
    train = datasets.Dataset.from_pandas(df)
    test = datasets.Dataset.from_pandas(dftest)
    dev = datasets.Dataset.from_pandas(dfdev)
    dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})

    return dataset

dataset = buildDatasets(train, dev, tests)

#build the model
model_name = "xlm-roberta-large" # we use the xlmr for tokenizing (large instead of base except for english)
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

# set training arguments
trainer_args = transformers.TrainingArguments(
    "../benchmark/checkpoints",
    evaluation_strategy="epoch",
    logging_strategy="epoch",   # changed all from steps
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    #eval_steps=100,
    #logging_steps=100,
    learning_rate=float(args.learning),# instead of 0.00001
    per_device_train_batch_size=args.batch, # instead of 8
    per_device_eval_batch_size=32,
    #max_steps=1500
)
data_collator = transformers.DataCollatorWithPadding(tokenizer)


#compute accuracy and loss
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels): 
    threshold=args.treshold
    # the treshold has to be really low because the probabilities of the predictions are not great,
    #find one that works best between 0.1 and 0.5 => 0.3 seems to be the best for now

    # sigmoid is used for multilabel, softmax for multiclass, argmax for binary
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