#!/bin/bash
#SBATCH --job-name=cache-test
#SBATCH --error=cache-test.err
#SBATCH --output=cache-test.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=16

./cache-test