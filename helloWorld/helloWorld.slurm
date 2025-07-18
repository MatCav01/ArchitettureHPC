#!/bin/bash
#SBATCH --job-name=helloWorld
#SBATCH --error=helloWorld.err
#SBATCH --output=helloWorld.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=16

./helloWorld