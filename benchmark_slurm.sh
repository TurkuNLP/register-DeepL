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
#srun cat monolingual_benchmark_files/train.tsv | python3 register-benchmark.py monolingual_benchmark_files/dev.tsv monolingual_benchmark_files/test.tsv
#srun zcat multilingual-register-data-new/en_train1.tsv.gz multilingual-register-data-new/en_train2.tsv.gz multilingual-register-data-new/en_train3.tsv.gz multilingual-register-data-new/en_train4.tsv.gz | python3 register-benchmark.py multilingual-register-data-new/en_dev.tsv.gz multilingual-register-data-new/en_test.tsv.gz
srun cat monolingual_benchmark_files/swe_train.tsv | python3 register-benchmark.py monolingual_benchmark_files/swe_dev.tsv monolingual_benchmark_files/swe_test.tsv --batch 7 --treshold 0.4 --epochs 6 --learning 0.00001
#srun cat monolingual_benchmark_files/fre_train.tsv | python3 register-benchmark.py monolingual_benchmark_files/fre_dev.tsv monolingual_benchmark_files/fre_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6
