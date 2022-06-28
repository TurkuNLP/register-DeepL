#!/bin/bash
#SBATCH --job-name=multilingual
#SBATCH --account=project_2005092 #2000539
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/small_languages/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# MULTI/CROSSLINGUAL (try on test set)

# FOR TRANSFER/TRANSLATED TESTING
FILES="test_sets/spa_test.tsv test_sets/pt_test_modified.tsv test_sets/jpn_test.tsv test_sets/chi_all.tsv"

# FOR ORIGINAL MULTILINGUAL TESTING/ DOWNSAMLED TESTING
#FILES="multilingual-register-data-new/formatted/en_test.formatted.tsv.gz multilingual-register-data-new/formatted/fi_test.formatted.tsv multilingual-register-data-new/formatted/fre_test.formatted.tsv multilingual-register-data-new/formatted/swe_test.formatted.tsv"



TRS=0.4 #"0.2 0.3 0.4 0.5 0.6 0.7 0.8"


for file in $FILES; do
for tr in $TRS; do
    echo $file
    echo $tr
    srun python3 register-code/multilabel/multilingual_evaluation.py --test_set data/{$file} --model saved_models/all_multilingual --treshold $tr
done
done



