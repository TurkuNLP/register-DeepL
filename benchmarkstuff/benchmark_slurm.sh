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
#FINNISH
#srun cat monolingual_benchmark_files/train.tsv | python3 register-benchmark-Fin_Eng.py monolingual_benchmark_files/dev.tsv monolingual_benchmark_files/test.tsv

#ENGLISH
#srun zcat multilingual-register-data-new/en_train1.tsv.gz multilingual-register-data-new/en_train2.tsv.gz multilingual-register-data-new/en_train3.tsv.gz multilingual-register-data-new/en_train4.tsv.gz | python3 register-benchmark-Fin_Eng.py multilingual-register-data-new/en_dev.tsv.gz multilingual-register-data-new/en_test.tsv.gz

#SWEDISH
#srun python3 register-benchmark-Fre_Swe.py monolingual_benchmark_files/Swe_train.tsv monolingual_benchmark_files/Swe_dev.tsv monolingual_benchmark_files/Swe_test.tsv --batch 7 --treshold 0.4 --epochs 6 --learning 1e-5

#FRENCH
srun python3 register-benchmark-Fre_Swe.py monolingual_benchmark_files/Fre_train.tsv monolingual_benchmark_files/Fre_dev.tsv monolingual_benchmark_files/Fre_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6
