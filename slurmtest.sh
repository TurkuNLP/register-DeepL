#!/bin/bash
#SBATCH --job-name=registertest
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=out/output%a.txt
#SBATCH --error=err/errors%a.txt

module load python/3.6
module load pytorch/1.11.0
srun python3 simple-register-test.py > output.txt

# for some reason this keeps failing at the beginning?
# probably some mistake here