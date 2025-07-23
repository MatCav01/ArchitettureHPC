#!/bin/bash

>results.out
exec > >(tee -a results.out)
# basedir=`basename $PWD`

echo "Serial version:"
grep dt: jacobi-serial.out
echo " "

echo "OpenMP version:"
grep dt: jacobi-omp.out
echo " "

echo "CUDA version:"
grep dt: jacobi-cuda.out
echo " "

echo "OpenMP target version:"
grep dt: jacobi-omptarget.out
echo " "

echo "MPI version:"
grep dt: jacobi-mpi.out
echo " "

echo "MPI + OpenMP version:"
grep dt: jacobi-mpiomp.out
echo " "

echo "MPI + CUDA version:"
grep dt: jacobi-mpicuda.out
echo " "

echo "MPI + OpenMP target version:"
grep dt: jacobi-mpiomptarget.out
echo " "
