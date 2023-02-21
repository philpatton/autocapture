#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00

#SBATCH --cpus-per-task=4
#SBATCH --mem=16000

#SBATCH --error=results/test-%A.err 
#SBATCH --output=results/test-%A.log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=pattonp@hawaii.edu

echo This is job $SLURM_JOB_ID running on $HOSTNAME

module load lang/Anaconda3

source activate mc

## python -m simulate --scenario test

python -m estimate --scenario test 
