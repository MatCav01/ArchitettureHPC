all: matrixMul-serial matrixMul-omp matrixMul-cuda

matrixMul-serial: matrixMul-serial.c
	gcc -Wall -O3 -fopenmp -o matrixMul-serial matrixMul-serial.c

matrixMul-omp: matrixMul-omp.c
	gcc -Wall -O3 -fopenmp -o matrixMul-omp matrixMul-omp.c

matrixMul-cuda: matrixMul-cuda.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o matrixMul-cuda matrixMul-cuda.cu

clean:
	/bin/rm matrixMul-serial matrixMul-omp matrixMul-cuda