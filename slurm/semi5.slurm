#!/bin/bash
#SBATCH --job-name=semi5
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=12
#SBATCH --mem=64000

#SBATCH --error=results/semi5-%A.err 
#SBATCH --output=results/semi5-%A.log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=pattonp@hawaii.edu

echo This is job $SLURM_JOB_ID running on $HOSTNAME

module load lang/Anaconda3

source activate mc

python -m estimate --scenario semi5