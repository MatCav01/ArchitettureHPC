all: saxpy-serial saxpy-omp saxpy-cuda saxpy-omptarget saxpy-mpi saxpy-mpiomp saxpy-mpicuda saxpy-mpiomptarget

saxpy-serial: saxpy-serial.c
	gcc -Wall -O3 -fopenmp -o saxpy-serial saxpy-serial.c

saxpy-omp: saxpy-omp.c
	gcc -Wall -O3 -fopenmp -o saxpy-omp saxpy-omp.c

saxpy-cuda: saxpy-cuda.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o saxpy-cuda saxpy-cuda.cu

saxpy-omptarget: saxpy-omptarget.c
	nvc -Wall -O3 -Minfo --ptxas-options=-v -mp=gpu -gpu=cc70 -fopenmp -o saxpy-omptarget saxpy-omptarget.c

saxpy-mpi: saxpy-mpi.c
	mpicc -Wall -O3 -fopenmp -o saxpy-mpi saxpy-mpi.c

saxpy-mpiomp: saxpy-mpiomp.c
	mpicc -Wall -O3 -fopenmp -o saxpy-mpiomp saxpy-mpiomp.c

saxpy-mpicuda: saxpy-mpicuda-kernel.cu saxpy-mpicuda.c
	nvcc -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o saxpy-mpicuda-kernel.o -c saxpy-mpicuda-kernel.cu
	mpicc -Wall -O3 -L${NVHPC_ROOT}/cuda/11.8/targets/x86_64-linux/lib -lmpi -lcudart -lcuda -fopenmp -o saxpy-mpicuda saxpy-mpicuda.c saxpy-mpicuda-kernel.o
	
saxpy-mpiomptarget: saxpy-mpiomptarget.c
	mpicc -Wall -O3 -Minfo -mp=gpu -gpu=cc70 -fopenmp -o saxpy-mpiomptarget saxpy-mpiomptarget.c

clean:
	rm saxpy-serial saxpy-omp saxpy-cuda saxpy-omptarget saxpy-mpi saxpy-mpiomp saxpy-mpicuda saxpy-mpiomptarget