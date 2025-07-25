#!/bin/bash

>results.out
exec > >(tee -a results.out)
# basedir=`basename $PWD`

echo "Serial version:"
grep statistics jacobi-serial.out
echo " "

echo "OpenMP version:"
grep statistics jacobi-omp.out
echo " "

echo "CUDA version:"
grep statistics jacobi-cuda.out
echo " "

echo "OpenMP target version:"
grep statistics jacobi-omptarget.out
echo " "

echo "MPI version:"
grep statistics jacobi-mpi.out
echo " "

echo "MPI + OpenMP version:"
grep statistics jacobi-mpiomp.out
echo " "

echo "MPI + CUDA version:"
grep statistics jacobi-mpicuda.out
echo " "

echo "MPI + OpenMP target version:"
grep statistics jacobi-mpiomptarget.out
echo " "
