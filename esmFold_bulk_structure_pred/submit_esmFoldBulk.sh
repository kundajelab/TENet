#!/bin/bash
#SBATCH --account=TODO
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32000M
#SBATCH --time=0-00:15
#SBATCH --array=1-1000 # TODO
#SBATCH -e stderr.txt
#SBATCH -o stdout.txt
#SBATCH --open-mode=append
#SBATCH --export=NONE

module load python/3.10
source $HOME/esm_model/bin/activate


python esmFold_bulk_pred.py $SLURM_ARRAY_TASK_ID