#!/bin/bash
#SBATCH --job-name=registertest
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=01:0:00 #30minutes for base, 1h for large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 
srun python3 register-multilabel.py main_labels_only/pt_FINAL.modified.tsv.gz test_sets/pt_test_modified.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6