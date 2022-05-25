#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err



module load pytorch 
srun zcat multilingual-register-data-new/en_train1.tsv.gz multilingual-register-data-new/en_train2.tsv.gz multilingual-register-data-new/en_train3.tsv.gz multilingual-register-data-new/en_train4.tsv.gz | python3 register-benchmark.py multilingual-register-data-new/en_dev.tsv.gz multilingual-register-data-new/en_test.tsv.gz
