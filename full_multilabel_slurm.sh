#!/bin/bash
#SBATCH --job-name=full_simplified
#SBATCH --account=project_2005092 # 2000539
#SBATCH --partition=gpu
#SBATCH --time=02:00:00 #1h 30 for 5 epochs, multi 5/6 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/full_simplified/%j.out
#SBATCH --error=../logs/%j.err


module load pytorch 


EPOCHS=6
LR=2e-5    # "1e-5 4e-6 5e-6 7e-5 8e-6"
TR=0.4    # "0.3 0.4 0.5 0.6"
BATCH=7


echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"


# PORTUGUESE
#srun python3 register-multilabel.py --train_set AfterDeepL/full_labels/pt_FINAL_full.tsv.gz --test_set test_sets/pt_test_modified.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/pt --lang pt --full

# SPANISH
#srun python3 register-multilabel.py --train_set AfterDeepL/full_labels/es_FINAL_full.tsv.gz --test_set test_sets/spa_test.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/spa --lang spa --full

#JAPANESE
#srun python3 register-multilabel.py --train_set AfterDeepL/full_labels/ja_FINAL_full.tsv.gz --test_set test_sets/jpn_test.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/jpn --lang jpn --full

#CHINESE
#srun python3 register-multilabel.py --train_set AfterDeepL/full_labels/chi_FINAL_full.tsv.gz --test_set test_sets/chi_all.tsv --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/chi --lang chi --full



# downsampled transfer

TEST="test_sets/pt_test_modified.tsv" #"test_sets/spa_test.tsv" #"test_sets/jpn_test.tsv" #"test_sets/chi_all.tsv" 

echo $TEST
srun python3 register-multilabel.py --train_set downsampled/full_labels/all_downsampled_full.tsv.gz --test_set $TEST --batch $BATCH --treshold $TR --epochs $EPOCHS --learning $LR --checkpoint ../multilabel/downsampled --lang downsampled_pt --full


echo "END: $(date)"