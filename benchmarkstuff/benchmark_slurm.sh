#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=project_2000539 # 2005092
#SBATCH --partition=gpu
#SBATCH --time=06:00:00 # 6 for english, 2 for anything else
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=../logs/%j.out
#SBATCH --error=../../logs/%j.err



module load pytorch 
#FINNISH
#srun cat monolingual_benchmark_files/train.tsv | python3 register-benchmark-Fin_Eng.py monolingual_benchmark_files/dev.tsv monolingual_benchmark_files/test.tsv

#ENGLISH
srun cat ../multilingual-register-data-new/formatted/en_train1.formatted.tsv ../multilingual-register-data-new/formatted/en_train2.formatted.tsv ../multilingual-register-data-new/formatted/en_train3.formatted.tsv ../multilingual-register-data-new/formatted/en_train4.formatted.tsv \
 | python3 register-benchmark-Fin_Eng.py ../multilingual-register-data-new/formatted/en_dev.formatted.tsv ../multilingual-register-data-new/formatted/en_test.formatted.tsv \
 --batch 7 \
 --treshold 0.4 \
 --epochs 5 \
 --learning 8e-6

/scratch/project_2005092/Anni/register-DeepL/multilingual-register-data-new/formatted/en_dev.formatted.tsv

#SWEDISH
#srun python3 register-benchmark-Fre_Swe.py monolingual_benchmark_files/Swe_train.tsv monolingual_benchmark_files/Swe_dev.tsv monolingual_benchmark_files/Swe_test.tsv --batch 7 --treshold 0.4 --epochs 6 --learning 1e-5

#FRENCH
#srun python3 register-benchmark-Fre_Swe.py monolingual_benchmark_files/Fre_train.tsv monolingual_benchmark_files/Fre_dev.tsv monolingual_benchmark_files/Fre_test.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6


# FINNISH FROM MULTILINGUAL DATASET
# run in benchmarkstuff directory
#srun python3 register-benchmark-Fre_Swe.py ../multilingual-register-data-new/formatted/fi_train.formatted.tsv ../multilingual-register-data-new/formatted/fi_dev.formatted.tsv ../multilingual-register-data-new/formatted/fi_test.formatted.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6
