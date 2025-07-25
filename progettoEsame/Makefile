all: jacobi-serial jacobi-omp jacobi-cuda jacobi-omptarget jacobi-mpi jacobi-mpiomp jacobi-mpicuda jacobi-mpiomptarget

FLAGS = -DGLX=${GLX} -DGLY=${GLY} -DMAXITER=${MAXITER} -DDUMP=${DUMP} -DDUMPSTEP=${DUMPSTEP}
CFLAGS = -Wall -O3 -fopenmp

jacobi-serial: common.h jacobi-serial.c
	gcc ${CFLAGS} ${FLAGS} -o jacobi-serial jacobi-serial.c

jacobi-omp: common.h jacobi-omp.c
	gcc ${CFLAGS} ${FLAGS} -o jacobi-omp jacobi-omp.c

jacobi-cuda: common.h jacobi-cuda.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35 ${FLAGS} -Xcompiler -fopenmp -o jacobi-cuda jacobi-cuda.cu

jacobi-omptarget: common.h jacobi-omptarget.c
	nvc ${CFLAGS} ${FLAGS} -Minfo --ptxas-options=-v -mp=gpu -gpu=cc70 -o jacobi-omptarget jacobi-omptarget.c

jacobi-mpi: common.h jacobi-mpi.c
	mpicc ${CFLAGS} ${FLAGS} -o jacobi-mpi jacobi-mpi.c

jacobi-mpiomp: common.h jacobi-mpiomp.c
	mpicc ${CFLAGS} ${FLAGS} -o jacobi-mpiomp jacobi-mpiomp.c

jacobi-mpicuda: common.h jacobi-mpicuda-kernel.cu jacobi-mpicuda.c
	nvcc -O3 -gencode arch=compute_35,code=sm_35 ${FLAGS} -Xcompiler -fopenmp -o jacobi-mpicuda-kernel.o -c jacobi-mpicuda-kernel.cu
	mpicc ${CFLAGS} ${FLAGS} -L${NVHPC_ROOT}/cuda/11.8/targets/x86_64-linux/lib -lmpi -lcudart -lcuda -o jacobi-mpicuda jacobi-mpicuda.c jacobi-mpicuda-kernel.o
	
jacobi-mpiomptarget: common.h jacobi-mpiomptarget.c
	mpicc ${CFLAGS} ${FLAGS} -Minfo -mp=gpu -gpu=cc70 -o jacobi-mpiomptarget jacobi-mpiomptarget.c

clean:
	rm -f jacobi-serial jacobi-omp jacobi-cuda jacobi-omptarget jacobi-mpi jacobi-mpiomp jacobi-mpicuda jacobi-mpiomptarget