#!/bin/bash
#SBATCH --job-name=saxpy-mpi
#SBATCH --output=saxpy-mpi.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun

module load devtoolset-11
module load openmpi/1.10.4

# compile here to optimize for the target architecture
make saxpy-mpi

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

export OMP_PLACES=cores     # place threads on hardware cores
export OMP_PROC_BIND=close  # place threads close each other

srun ./saxpy-mpi
