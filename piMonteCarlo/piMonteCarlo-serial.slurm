#!/bin/bash
#SBATCH --job-name=piMonteCarlo-serial
#SBATCH --output=piMonteCarlo-serial.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=shortrun

module load devtoolset-11
module load cuda
module load openmpi

make piMonteCarlo-serial

srun ./piMonteCarlo-serial

echo " "

sacct -j $SLURM_JOB_ID.0 --format=JobID%16,AllocNodes,NTasks,Elapsed,ConsumedEnergy


echo " "
echo " "
