#!/bin/bash
#SBATCH --job-name=translated
#SBATCH --account=project_2005092 # 2000539
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


EPOCHS=3 #5
LR=5e-6    # "1e-5 4e-6 5e-6 7e-5 8e-6"
TR=0.3    # "0.3 0.4 0.5 0.6"
BATCH=7


echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"


# PORTUGUESE
#srun python3 register-multilabel.py --train_set AfterDeepL/main_labels_only/pt_FINAL.modified.tsv.gz --test_set test_sets/pt_test_modified.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/pt --lang pt

# SPANISH
#srun python3 register-multilabel.py --train_set AfterDeepL/main_labels_only/es_FINAL.modified.tsv.gz --test_set test_sets/spa_test.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/spa --lang spa

#JAPANESE
#srun python3 register-multilabel.py --train_set AfterDeepL/main_labels_only/ja_FINAL.modified.tsv.gz --test_set test_sets/jpn_test.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/jpn --lang jpn

#CHINESE
srun python3 register-multilabel.py --train_set AfterDeepL/main_labels_only/chi_FINAL.modified.tsv.gz --test_set test_sets/chi_all.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/chi --lang chi


#FINNISH TEST
# srun python3 register-multilabel.py --train_set AfterDeepL/main_labels_only/FIN_FINAL.modified.tsv.gz --test_set old-datasets/multilingual-register-data-new/main_labels_only/fi_test.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/fin --lang fin

# transfer test for finnish with eng, fre, swe downsampled sets (= same as translated)
# srun python3 register-multilabel.py --train_set downsampled/main_labels_only/en_train.downsampled_modified.tsv downsampled/main_labels_only/fre_train.downsampled_modified.tsv downsampled/main_labels_only/swe_train.downsampled_modified.tsv --test_set old-datasets/multilingual-register-data-new/main_labels_only/fi_test.tsv \
# --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/fin --lang fintransfer



# I think the multilingual model for the translations is quite questionable 
#since it uses the same data 4 four times just translated to different languages


# MULTI/CROSSLINGUAL (all translated files and original downsampled as train and dev)

# srun python3 register-multilabel.py --train_set AfterDeepL/chi_FINAL.tsv.gz \
#  AfterDeepL/ja_FINAL.tsv.gz AfterDeepL/es_FINAL.tsv.gz AfterDeepL/pt_FINAL.tsv.gz \
#  downsampled/all_downsampled.tsv.gz \
#  --test_set test_sets/chi_all.tsv \
#  --batch $BATCH --treshold $TR --epochs $EPOCHS \
#  --learning $LR --multilingual --checkpoint ../multilabel/all --saved saved_models/all_multilingual


# this will take a long time, set to 5/6 hours + 1/2 hours because of original downsampled
# the test set is just for show but is not used