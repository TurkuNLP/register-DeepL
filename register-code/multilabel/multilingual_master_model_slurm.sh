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

PATH="data/old-datasets/multilingual-register-data-new/formatted/"

srun python3 multilingual_master_model.py \
    --train_sets {$PATH}en_train1.formatted.tsv \
    {$PATH}en_train2.formatted.tsv \
    {$PATH}en_train3.formatted.tsv \
    {$PATH}en_train4.formatted.tsv \
    {$PATH}fi_train.formatted.tsv \
    {$PATH}fre_train.formatted.tsv \
    {$PATH}swe_train.formatted.tsv \
    --dev_sets {$PATH}en_dev.formatted.tsv \
    {$PATH}fi_dev.formatted.tsv \
    {$PATH}fre_dev.formatted.tsv \
    {$PATH}swe_dev.formatted.tsv \
    # --test_sets multilingual-register-data-new/formatted/en_test.formatted.tsv \
    # multilingual-register-data-new/formatted/fi_test.formatted.tsv \
    # multilingual-register-data-new/formatted/fre_test.formatted.tsv \
    # multilingual-register-data-new/formatted/swe_test.formatted.tsv \
    --batch 7 \
    --treshold 0.4 \
    --epochs 5 \
    --learning 8e-6 \



