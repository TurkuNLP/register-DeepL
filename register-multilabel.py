from random import shuffle
import transformers
import datasets
import gzip 
import torch
import argparse
from pprint import PrettyPrinter
import logging
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

    return args 

args = arguments()

# manual list of the main labels
unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]

train = datasets.load_dataset(
    "csv", 
    data_files={'train':args.train_set}, 
    delimiter="\t",
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
      "label":datasets.Value("string")})
    )

# remember to shuffle because the data is in en,fi,fre,swe order!
# this shuffles by default
train, dev = train.train_test_split(test_size=0.2).values()

test = datasets.load_dataset(
    "csv", 
    data_files={'test':args.test_set}, 
    delimiter="\t",
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
      "label":datasets.Value("string")})
    )

dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})
pprint(dataset)

# the data is fitted to these main labels
unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]

def split_labels(dataset):
    # for some reason NA ends up as None in the dataset??? at least in some cases
    if dataset['label'] == None:
        dataset['label'] = np.array('NA')
    else:
        dataset['label'] = np.array(dataset['label'].split())
    return dataset

def binarize(dataset):
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_labels])
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])[0]})
    return dataset

pprint(dataset['train']['label'][:5])
dataset = dataset.map(split_labels)
pprint(dataset['train']['label'][:5])
dataset = binarize(dataset)
pprint(dataset['train']['label'][:5])

# then use the XLMR tokenizer
model_name = "xlm-roberta-large"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True # use some other method for this?
    )

dataset = dataset.map(tokenize)

num_labels = len(unique_labels)
model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

trainer_args = transformers.TrainingArguments(
    "../multilabel/checkpoints", # change this to put the checkpoints somewhere else
    evaluation_strategy="epoch",
    logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    learning_rate=args.learning,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)

#compute accuracy and loss
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels):
    # the treshold has to be really low because the probabilities of the predictions are not great, could even do without any treshold then? or find one that works best between 0.1 and 0.5
    threshold=args.treshold

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