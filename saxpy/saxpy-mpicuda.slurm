#!/bin/bash
#SBATCH --job-name=saxpy-mpicuda
#SBATCH --output=saxpy-mpicuda.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:16

module load devtoolset-11
module load openmpi/1.10.4
module load nvhpc-nompi/24.7

# compile here to optimize for the target architecture
make saxpy-mpicuda

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

srun saxpy-mpicuda
