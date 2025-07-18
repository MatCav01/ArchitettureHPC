#!/bin/bash
#SBATCH --job-name=piMonteCarlo-mpi
#SBATCH --output=piMonteCarlo-mpi.out
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=shortrun

module load devtoolset-11
module load cuda
module load openmpi

make piMonteCarlo-mpi

srun ./piMonteCarlo-mpi

echo " "

sacct -j $SLURM_JOB_ID.0 --format=JobID%16,AllocNodes,NTasks,Elapsed,ConsumedEnergy


echo " "
echo " "
