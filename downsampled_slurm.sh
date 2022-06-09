#!/bin/bash
#SBATCH --job-name=downsampled
#SBATCH --account=project_2005092
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
# original test set had empty texts but that is now fixed
#srun python3 register-multilabel.py main_labels_only/original_downsampled/en_train.downsampled_modified.tsv multilingual-register-data-new/formatted/en_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FINNISH
#srun python3 register-multilabel.py main_labels_only/original_downsampled/fi_train.downsampled_modified.tsv multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# SWEDISH
#srun python3 register-multilabel.py main_labels_only/original_downsampled/swe_train.downsampled_modified.tsv multilingual-register-data-new/formatted/swe_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FRENCH WAS NOT DOWNSAMPLED (just shuffled) SO NO NEED FOR THAT


# TRY MULTILINGUAL DOWNSAMPLED MODEL AS WELL
# => not enough disk space, change sbatch stuff, give more time too
#OSError: Not enough disk space. Needed: Unknown size (download: Unknown size, generated: Unknown size, post-processed: Unknown size)
# => tried emptying cache => NOW IT WORKS
# ENGLISH TEST
#srun python3 register-multilabel.py main_labels_only/original_downsampled/all_downsampled.tsv.gz multilingual-register-data-new/formatted/en_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FINNISH TEST
#srun python3 register-multilabel.py main_labels_only/original_downsampled/all_downsampled.tsv.gz multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# SWEDISH TEST
#srun python3 register-multilabel.py main_labels_only/original_downsampled/all_downsampled.tsv.gz multilingual-register-data-new/formatted/swe_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6

# FRENCH TEST
srun python3 register-multilabel.py main_labels_only/original_downsampled/all_downsampled.tsv.gz multilingual-register-data-new/formatted/fre_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6
# here is still some problem with empty text in the test set


#WITH COMMON TEST SET? what about using dev sets?
# now I just split data to train and dev