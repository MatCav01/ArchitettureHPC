#!/bin/bash
#SBATCH --job-name=saxpy-mpiomptarget
#SBATCH --output=saxpy-mpiomptarget.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=skyvolta
#SBATCH --nodelist=node06
#SBATCH --gres=gpu:v100

unset CUDA_VISIBLE_DEVICES

module load devtoolset-11
module load openmpi/1.10.4
module load nvhpc-nompi/24.7

export OMPI_CC=nvc

# compile here to optimize for the target architecture
make saxpy-mpiomptarget

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

# export OMP_PLACES=cores     # place threads on hardware cores
# export OMP_PROC_BIND=close  # place threads close each other

srun ./saxpy-mpiomptarget
