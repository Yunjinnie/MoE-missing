#!/bin/bash

#SBATCH --job-name=mosi                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 12 hour timelimit
#SBATCH --mem=200000MB                         # Using 10GB CPU Memory
#SBATCH --partition=P2                        # Using "b" partition 
#SBATCH --cpus-per-task=16                     # Using 4 maximum processor

# CMUMOSI

#export HOME
#source /anaconda3/bin/activate
source /home/s2/yunjinna/.bashrc
source /home/s2/yunjinna/anaconda/bin/activate
conda activate env2

srun python -u main.py