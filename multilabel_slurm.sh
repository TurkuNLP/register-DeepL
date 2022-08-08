#!/bin/bash
#SBATCH --job-name=translated
#SBATCH --account=project_2005092 #2005092 # 2000539
#SBATCH --partition=gpu
#SBATCH --time=02:00:00 #1h 30 for 5 epochs, multi 5/6 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/small_languages/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# # with current settings set time to 16 x 1.30 = 24

# EPOCHS=5 ##{5..7..1} # last one is increment
# LR="4e-6 5e-6 7e-5 8e-6" # learning rate
# BATCH=8 #"7 8"
# TR="0.3 0.4 0.5 0.6" #treshold
# # $I={1..2..1} also loop so that it reproduces the same thing a couple of times?


# TRAIN=main_labels_only/ja_FINAL.modified.tsv.gz
# TEST=test_sets/jpn_test_modified.tsv

# echo "train: $TRAIN test: $TEST"

# # for BATCH in $BATCH; do
# # for EPOCHS in $EPOCHS_; do
# for rate in $LR; do
# for treshold in $TR; do
# echo "learning rate: $rate treshold: $treshold batch: $BATCH epochs: $EPOCHS"
# srun python3 register-multilabel.py \
#     --train_set $TRAIN \
#     --test_set $TEST \
#     --batch $BATCH \
#     --treshold $treshold \
#     --epochs $EPOCHS \
#     --learning $rate
# done
# done
# done
# done


EPOCHS=5 #5
LR=8e-6    # "1e-5 4e-6 5e-6 7e-5 8e-6"
TR=0.4    # "0.3 0.4 0.5 0.6"
BATCH=8
DATAPATH="data/AfterDeepL/main_labels_only/" # for some reason this breaks the script atm and makes it not find srun command, PATH is already a variable I am overwriting?
MODEL="xlm-roberta-large" #'TurkuNLP/bert-base-finnish-cased-v1' #"xlm-roberta-large"
# PT 'neuralmind/bert-large-portuguese-cased'
# ZH 'hfl/chinese-macbert-base'

echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"

# PORTUGUESE
# srun python3 register-multilabel.py --train_set ${DATAPATH}pt_FINAL.modified.tsv.gz --test_set data/test_sets/main_labels_only/pt_test_modified.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/pt --lang pt --model $MODEL

# SPANISH
# srun python3 register-multilabel.py --train_set ${DATAPATH}es_FINAL.modified.tsv.gz --test_set data/test_sets/main_labels_only/spa_test_modified.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/spa --lang spa --model $MODEL 

#JAPANESE
# srun python3 register-multilabel.py --train_set ${DATAPATH}ja_FINAL.modified.tsv.gz --test_set data/test_sets/main_labels_only/jpn_test_modified.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/jpn --lang jpn --model $MODEL

#CHINESE
srun python3 register-multilabel.py --train_set ${DATAPATH}chi_FINAL.modified.tsv.gz --test_set data/test_sets/main_labels_only/chi_all_modified.tsv \
--batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/chi --lang chi --model $MODEL


#TRANSFER

# transfer test for some language with eng, fre, swe downsampled sets (= same as translated)
# srun python3 register-multilabel.py --train_set data/downsampled/main_labels_only/all_downsampled.tsv.gz --test_set data/test_sets/main_labels_only/chi_all_modified.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/transfernodev3 --lang transfer




#FINNISH TEST
# srun python3 register-multilabel.py --train_set ${DATAPATH}FIN_FINAL.modified.tsv.gz --test_set data/old-datasets/multilingual-register-data-new/main_labels_only/fi_test.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/fin --lang fin --model $MODEL

# transfer test for finnish with eng, fre, swe downsampled sets (= same as translated)
# srun python3 register-multilabel.py --train_set data/downsampled/main_labels_only/all_downsampled.tsv.gz --test_set data/old-datasets/multilingual-register-data-new/main_labels_only/fi_test.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/fin --lang fintransfer

#SWEDISH TEST
# srun python3 register-multilabel.py --train_set ${DATAPATH}SWE_FINAL.modified.tsv.gz --test_set data/old-datasets/multilingual-register-data-new/main_labels_only/swe_test.formatted_modified.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/swe --lang swe

#Swedish transfer test
# srun python3 register-multilabel.py --train_set data/downsampled/main_labels_only/all_downsampled.tsv.gz --test_set data/old-datasets/multilingual-register-data-new/main_labels_only/swe_test.formatted_modified.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/swe --lang swetransfer



# I think the multilingual model for the translations is quite questionable 
#since it uses the same data 4 four times just translated to different languages


# MULTI/CROSSLINGUAL (all translated files and original downsampled as train and dev)

# srun python3 register-multilabel.py --train_set ${DATAPATH}chi_FINAL.tsv.gz \
#  ${DATAPATH}ja_FINAL.tsv.gz AfterDeepL/es_FINAL.tsv.gz ${DATAPATH}pt_FINAL.tsv.gz \
#  data/downsampled/main_labels_only/all_downsampled.tsv.gz  \
#  --test_set test_sets/chi_all.tsv \
#  --batch $BATCH --treshold $TR --epochs $EPOCHS \
#  --learning $LR --multilingual --checkpoint ../multilabel/all --saved saved_models/all_multilingual


# this will take a long time, set to 5/6 hours + 1/2 hours because of original downsampled
# the test set is just for show but is not used


echo "END: $(date)"