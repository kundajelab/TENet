#!/bin/bash
#SBATCH --account=TODO # Please enter appropriate slurm account details
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80000M
#SBATCH --time=1-00:00

module load python/3.10
source $HOME/esm_model/bin/activate


python model.py
python eval_model.py