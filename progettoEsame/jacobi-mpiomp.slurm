#!/bin/bash
#SBATCH --job-name=jacobi-mpiomp
#SBATCH --output=jacobi-mpiomp.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun

module load devtoolset-11
module load openmpi/1.10.4

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
make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-mpiomp

if [[ $? != 0 ]];
then
    echo "compilation FAILED!"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

export OMP_PLACES=cores     # place threads on hardware cores
export OMP_PROC_BIND=close  # place threads close each other

srun ./jacobi-mpiomp
