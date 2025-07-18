#!/bin/bash
#SBATCH --job-name=matrix
#SBATCH --error=matrix.err
#SBATCH --output=matrix.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=16

./matrix