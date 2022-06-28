# Register labeling code and data: 


This repository includes code for register labeling (multi-label classification) using huggingface transformers and datasets and pytorch as well as code for making files for [translation](codedump/translation_pipeline/) and all the used data: originals, downsampled and translated.

The goal is to compare cross-lingual transfer using XLM-Roberta (from en, fi, swe, fre to small languages pt, spa, jp, zh) and translation using DeepL to the small languages using XLM-Roberta and monolingual models. 

It is possible to use either only upper labels (8) which tells the other upper category to which the text belongs to or full simplified labels (24) which use sublabels to further identify the genre/register of the text.

## Table for the different tests:

| Source  | Target |
| ------------- | ------------- |
| en, fi, swe, fre  | pt  |
| en, fi, swe, fre  | spa  |
| en, fi, swe, fre  | jp  |
| en, fi, swe, fre  | zh  |
| pt  | pt  |
| spa  | spa  |
| jp  | jp  |
| zh  | zh  |


## How to use the script:
register-multilabel.py can be used for example as follows:

```
python3 register-multilabel.py --train_set [FILE/FILES] --test_set [FILE/FILES] [--full] --batch [NUM] --treshold [NUM] --epochs [NUM] --learning [NUM] --checkpoint [PATH] --model [MODEL] 
```
There are a few other arguments that can be used but only --train_set and --test_set are required. To get more information about the possible arguments, try using the -h flag:

```
python3 register-multilabel.py -h
```