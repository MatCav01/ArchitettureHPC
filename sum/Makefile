all: sum-cuda

sum-cuda: sum-cuda.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o sum-cuda sum-cuda.cu

clean:
	/bin/rm sum-cuda