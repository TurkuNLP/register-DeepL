#!/bin/bash
#SBATCH --job-name=multilingual
#SBATCH --account=project_2000539 #2005092
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/small_languages/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# MULTI/CROSSLINGUAL (try on test set)

# more loops to test all kings oflearning rates, epochs, tresholds ets.

# FOR TRANSFER/TRANSLATED TESTING
FILES="test_sets/spa_test_modified.tsv test_sets/pt_test_modified.tsv test_sets/jpn_test_modified.tsv test_sets/chi_all_modified.tsv"

# FOR ORIGINAL MULTILINGUAL TESTING
#FILES="multilingual-register-data-new/formatted/en_test.formatted.tsv multilingual-register-data-new/formatted/fi_test.formatted.tsv multilingual-register-data-new/formatted/fre_test.formatted.tsv multilingual-register-data-new/formatted/swe_test.formatted.tsv"

# FOR DOWNSAMPLED TESTING




for file in $FILES; do
    srun python3 multilingual_evaluation.py --test_set $file --model saved_models/downsampled_multilingual
done





# SPANISH
#srun python3 multilingual_evaluation.py test_sets/spa_test_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6

# PORTUGUESE
#srun python3 multilingual_evaluation.py test_sets/pt_test_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6

# JAPANESE
#srun python3 multilingual_evaluation.py test_sets/jpn_test_modified.tsv --batch 8 --treshold 0.4 --epochs 5 --learning 8e-6

# CHINESE
#srun python3 multilingual_evaluation.py test_sets/chi_all_modified.tsv --batch 7 --treshold 0.4 --epochs 5 --learning 8e-6



# ALL ORIGINAL DATASETS THE SAME WAY AS ABOVE?

