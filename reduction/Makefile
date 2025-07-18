all: reduction-serial reduction-omp reduction-cuda

reduction-serial: reduction-serial.c
	gcc -Wall -O3 -fopenmp -o reduction-serial reduction-serial.c

reduction-omp: reduction-omp.c
	gcc -Wall -O3 -fopenmp -o reduction-omp reduction-omp.c

reduction-cuda: reduction-cuda.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o reduction-cuda reduction-cuda.cu

clean:
	/bin/rm reduction-serial reduction-omp reduction-cuda