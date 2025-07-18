#!/bin/bash
#SBATCH --job-name=saxpy-serial
#SBATCH --output=saxpy-serial.out
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
make saxpy-serial

if [[ $? != 0 ]];
then
    echo "compilation FAILED !"
    exit -1
else
    echo "compilation SUCCEEDED."
fi

export OMP_PLACES=cores     # place threads on hardware cores
export OMP_PROC_BIND=close  # place threads close each other

srun ./saxpy-serial
