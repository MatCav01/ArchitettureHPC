#!/bin/bash
#SBATCH --job-name=reduction-cuda
#SBATCH --output=reduction-cuda.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1

module load devtoolset-11
module load nvhpc/24.7

# compile here to optimize for the target architecture
make reduction-cuda

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

./reduction-cuda
