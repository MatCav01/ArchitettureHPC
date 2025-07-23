#!/bin/bash

>results.out
exec > >(tee -a results.out)
basedir=`basename $PWD`

echo "Serial version:"
grep dt: $basedir-serial.out
echo " "

echo "OpenMP version:"
grep dt: $basedir-omp.out
echo " "

echo "CUDA version:"
grep dt: $basedir-cuda.out
echo " "

echo "OpenMP target version:"
grep dt: $basedir-omptarget.out
echo " "

echo "MPI version:"
grep dt: $basedir-mpi.out
echo " "

echo "MPI + OpenMP version:"
grep dt: $basedir-mpiomp.out
echo " "

echo "MPI + CUDA version:"
grep dt: $basedir-mpicuda.out
echo " "

echo "MPI + OpenMP target version:"
grep dt: $basedir-mpiomptarget.out
echo " "
