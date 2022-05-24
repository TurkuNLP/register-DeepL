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


# export SOURCE_DIR=/scratch/project_2005092/Anni/register-DeepL
# export FILE_DIR=/scratch/project_2005092/Anni/register-DeepL/AfterDeepL

#module load python/3.6
module load pytorch #/1.11.0 # I guess this did the trick, I did not have this specific version?
srun python3 simple-register-with-tests.py main_labels_only/es_FINAL.modified.tsv.gz test_sets/spa_test_modified.tsv #downsampled/en_train.downsampled.tsv 
