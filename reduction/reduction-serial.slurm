#!/bin/bash
#SBATCH --job-name=reduction-serial
#SBATCH --output=reduction-serial.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun

module load devtoolset-11

# compile here to optimize for the target architecture
make reduction-serial

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

./reduction-serial
