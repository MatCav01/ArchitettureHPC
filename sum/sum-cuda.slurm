#!/bin/bash
#SBATCH --job-name=sum-cuda
#SBATCH --output=sum-cuda.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1

module load devtoolset-11
module load nvhpc/24.7

# compile here to optimize for the target architecture
make sum-cuda

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

export OMP_PLACES=cores     # place threads on hardware cores
export OMP_PROC_BIND=close  # place threads close each other

# run and redirect standard-output to file data.dat
srun ./sum-cuda
