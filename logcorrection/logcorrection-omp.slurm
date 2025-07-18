#!/bin/bash
#SBATCH --job-name=logcorrection-omp
#SBATCH --output=logcorrection-omp-%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH --partition=shortrun

module load devtoolset-11

# compile here to optimize for the target architecture
make logcorrection-omp

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
srun ./logcorrection-omp | tee data.dat
