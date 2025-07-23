#!/bin/bash
#SBATCH --job-name=jacobi-omp
#SBATCH --output=jacobi-omp.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=skyvolta

module load devtoolset-11

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
make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-omp

if [[ $? != 0 ]];
then
    echo "compilation FAILED!"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

export OMP_PLACES=cores     # place threads on hardware cores
export OMP_PROC_BIND=close  # place threads close each other

srun ./jacobi-omp
