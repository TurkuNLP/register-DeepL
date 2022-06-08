import transformers
import datasets
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

    return args 

args = arguments()

dataset = datasets.load_dataset(
    "csv", 
    data_files={'train':args.train_set, 'test':args.test_set, 'dev': args.dev_set}, 
    delimiter="\t",
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
      "label":datasets.Value("string")})
    )

# the data is fitted to these main labels
unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]

def split_labels(dataset):
    # for some reason one NA ends up as None in the dataset???
    if dataset['label'] == None:
        dataset['label'] = np.array('NA')
    else:
        dataset['label'] = np.array(dataset['label'].split())
    # do I have to do something else?
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