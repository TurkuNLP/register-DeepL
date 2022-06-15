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

BATCH=8
LR=8e-6
TR=0.4
EPOCHS=5

echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"

#downsampled main labels only

# ENGLISH
#srun python3 register-multilabel.py --train_set downsampled/en_train.downsampled.tsv --test_set multilingual-register-data-new/formatted/en_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FINNISH
#srun python3 register-multilabel.py --train_set downsampled/fi_train.downsampled.tsv --test_set multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# SWEDISH
#srun python3 register-multilabel.py --train_set downsampled/swe_train.downsampled.tsv --test_set multilingual-register-data-new/formatted/swe_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FRENCH WAS NOT DOWNSAMPLED (just shuffled) SO NO NEED FOR THAT
# running anyway because the dev set was taken out
#srun python3 register-multilabel.py --train_set downsampled/fre_train.downsampled.tsv --test_set multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6


# TRY MULTILINGUAL DOWNSAMPLED MODEL AS WELL

#this does not work??
#srun python3 register-multilabel.py --train_set downsampled/en_train.downsampled.tsv downsampled/fi_train.downsampled.tsv downsampled/swe_train.downsampled.tsv downsampled/fre_train.downsampled.tsv --test_set multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --multilingual --checkpoint ../multilabel/downsampled --saved saved_models/downsampled_multilingual
# this works
#srun python3 register-multilabel.py --train_set downsampled/all_downsampled.tsv.gz --test_set multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --multilingual --checkpoint ../multilabel/downsampled --saved saved_models/downsampled_multilingual

# TEST TO SEE HOW THE TRANSFER EVAL GOES

TEST="test_sets/pt_test_modified.tsv" #"test_sets/spa_test.tsv" #"test_sets/jpn_test.tsv" #"test_sets/chi_all.tsv" 

echo $TEST
srun python3 register-multilabel.py --train_set downsampled/all_downsampled.tsv.gz --test_set $TEST --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/downsampled --lang downsampled_pt
