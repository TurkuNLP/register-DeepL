#!/bin/bash
#SBATCH --job-name=translated
#SBATCH --account=project_2000539 #2005092
#SBATCH --partition=gpu
#SBATCH --time=24:00:00 #1h 30 for 7 epochs, multi 5/6 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/small_languages/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# with current settings set time to 16 x 1.30 = 24

EPOCHS=5 ##{5..7..1} # last one is increment
LR="4e-6 5e-6 7e-5 8e-6" # learning rate
BATCH=8 #"7 8"
TR="0.3 0.4 0.5 0.6" #treshold
# $I={1..2..1} also loop so that it reproduces the same thing a couple of times?


TRAIN=main_labels_only/chi_FINAL.modified.tsv.gz
TEST=test_sets/chi_all_modified.tsv


# for BATCH in $BATCH; do
# for EPOCHS in $EPOCHS_; do
for rate in $LR; do
for treshold in $TR; do
echo "learning rate: $rate treshold: $treshold batch: $BATCH epochs: $EPOCHS"
srun python3 register-multilabel.py \
    --train_set $TRAIN \
    --test_set $TEST \
    --batch $BATCH \
    --treshold $treshold \
    --epochs $EPOCHS \
    --learning $rate
done
done
# done
# done

# PORTUGUESE
#srun python3 register-multilabel.py --train_set main_labels_only/pt_FINAL.modified.tsv.gz --test_set test_sets/pt_test_modified.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 4e-6

# SPANISH
#srun python3 register-multilabel.py --train_set main_labels_only/es_FINAL.modified.tsv.gz --test_set test_sets/spa_test_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6

#JAPANESE
#srun python3 register-multilabel.py --train_set main_labels_only/ja_FINAL.modified.tsv.gz --test_set test_sets/jpn_test_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6

#CHINESE
#srun python3 register-multilabel.py --train_set main_labels_only/chi_FINAL.modified.tsv.gz --test_set test_sets/chi_all_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6




# I think the multilingual model for the translations is quite questionable 
#since it uses the same data 4 four times just translated to different languages


# MULTI/CROSSLINGUAL (all translated files as train and dev)
# common train file 

#srun python3 register-multilabel.py --train_set main_labels_only/all_translated.tsv.gz --test_set test_sets/chi_all_modified.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6 --multilingual
# this will take a long time, set to 5/6 hours
# the test set is just for show but is not used

# for the multilingual I could also make the dev set be all of the test files?
# or just pick one to make specialized multilingual models