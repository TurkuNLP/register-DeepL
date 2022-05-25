#!/bin/bash
#SBATCH --job-name=registertest
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=01:0:00 #from 2h to 30minutes for base, 1h for large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 
srun python3 register-multilabel.py main_labels_only/pt_FINAL.modified.tsv.gz test_sets/pt_test_modified.tsv #downsampled/en_train.downsampled.tsv 

# should try with the downsampled in a multilabel setting to get an english baseline
# (same as with the original english in a multilabel setting that I ran)