#!/bin/bash
#SBATCH --job-name=downsampled
#SBATCH --account=project_2005092 #2000539
#SBATCH --partition=gpu
#SBATCH --time=01:30:00 #1h 30 for multi, 30min for downsampled
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/downsampled_tests/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

BATCH=7
LR=8e-6
TR=0.4
EPOCHS=5
MODEL='xlm-roberta-large'

echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"

#downsampled main labels only

# ENGLISH
#srun python3 register-multilabel.py --train_set data/downsampled/en_train.downsampled.tsv --test_set data/old-datasets/multilingual-register-data-new/formatted/en_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FINNISH
#srun python3 register-multilabel.py --train_set data/downsampled/fi_train.downsampled.tsv --test_set data/old-datasets/multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# SWEDISH
#srun python3 register-multilabel.py --train_set data/downsampled/swe_train.downsampled.tsv --test_set data/old-datasets/multilingual-register-data-new/formatted/swe_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FRENCH WAS NOT DOWNSAMPLED (just shuffled) SO NO NEED FOR THAT
# running anyway because the dev set was taken out
#srun python3 register-multilabel.py --train_set data/downsampled/fre_train.downsampled.tsv --test_set data/old-datasets/multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6


# TRY MULTILINGUAL DOWNSAMPLED MODEL AS WELL

TEST="test_sets/main_labels_only/pt_test_modified.tsv" #"test_sets/main_labels_only/spa_test_modified.tsv" #"test_sets/main_labels_only/jpn_test_modified.tsv" #"test_sets/main_labels_only/chi_all_modified.tsv" 

echo $TEST
srun python3 register-multilabel.py --train_set data/downsampled/main_labels_only/all_downsampled.tsv.gz --test_set data/$TEST --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/downsampled --lang downsampled --model $MODEL #--multilingual --saved saved_models/downsampled_multilingual
