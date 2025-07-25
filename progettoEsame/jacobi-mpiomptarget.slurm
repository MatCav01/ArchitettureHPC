#!/bin/bash
#SBATCH --job-name=jacobi-mpiomptarget
#SBATCH --output=jacobi-mpiomptarget.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=skyvolta
#SBATCH --nodelist=node06
#SBATCH --gres=gpu:v100

unset CUDA_VISIBLE_DEVICES

module load devtoolset-11
module load openmpi/1.10.4
module load nvhpc/24.7

export OMPI_CC=nvc

GLX=512
GLY=512
MAXITER=10000
DUMP=0
DUMPSTEP=500

if [[ ! -d video ]];
then
    echo "creating directory video..."
    mkdir video
fi

# compile here to optimize for the target architecture
make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-mpiomptarget

if [[ $? != 0 ]];
then
    echo "compilation FAILED!"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

srun ./jacobi-mpiomptarget
