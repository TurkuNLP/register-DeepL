#!/bin/bash
#SBATCH --job-name=downsampled
#SBATCH --account=project_2000539 #2005092
#SBATCH --partition=gpu
#SBATCH --time=01:30:00 #1h 30 for multi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/downsampled_tests/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

#downsampled main labels only

# ENGLISH
#srun python3 register-multilabel.py --train_set main_labels_only/original_downsampled/en_train.downsampled_modified.tsv --test_set multilingual-register-data-new/formatted/en_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FINNISH
#srun python3 register-multilabel.py --train_set main_labels_only/original_downsampled/fi_train.downsampled_modified.tsv --test_set multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# SWEDISH
#srun python3 register-multilabel.py --train_set main_labels_only/original_downsampled/swe_train.downsampled_modified.tsv --test_set multilingual-register-data-new/formatted/swe_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FRENCH WAS NOT DOWNSAMPLED (just shuffled) SO NO NEED FOR THAT
# running anyway because the dev set was taken out
srun python3 register-multilabel.py --train_set main_labels_only/original_downsampled/fre_train.downsampled_modified.tsv --test_set multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6


# TRY MULTILINGUAL DOWNSAMPLED MODEL AS WELL

#srun python3 register-multilabel.py --train_set main_labels_only/original_downsampled/all_downsampled.tsv.gz --test_set multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6 --multilingual
