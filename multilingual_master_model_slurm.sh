#!/bin/bash
#SBATCH --job-name=mastermodel
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=06:00:00 # how long is needed? 5/6  hours apparently
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# MULTI/CROSSLINGUAL (all original files)

srun python3 multilingual_master_model.py \
    --train_sets multilingual-register-data-new/formatted/en_train1.formatted.tsv \
    multilingual-register-data-new/formatted/en_train2.formatted.tsv \
    multilingual-register-data-new/formatted/en_train3.formatted.tsv \
    multilingual-register-data-new/formatted/en_train4.formatted.tsv \
    multilingual-register-data-new/formatted/fi_train.formatted.tsv \
    multilingual-register-data-new/formatted/fre_train.formatted.tsv \
    multilingual-register-data-new/formatted/swe_train.formatted.tsv \
    --dev_sets multilingual-register-data-new/formatted/en_dev.formatted.tsv \
    multilingual-register-data-new/formatted/fi_dev.formatted.tsv \
    multilingual-register-data-new/formatted/fre_dev.formatted.tsv \
    multilingual-register-data-new/formatted/swe_dev.formatted.tsv \
    # --test_sets multilingual-register-data-new/formatted/en_test.formatted.tsv \
    # multilingual-register-data-new/formatted/fi_test.formatted.tsv \
    # multilingual-register-data-new/formatted/fre_test.formatted.tsv \
    # multilingual-register-data-new/formatted/swe_test.formatted.tsv \
    --batch 7 \
    --treshold 0.4 \
    --epochs 5 \
    --learning 8e-6 \



