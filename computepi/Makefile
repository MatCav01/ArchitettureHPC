all: computepi-serial computepi-omp

computepi-serial: computepi-serial.c
	gcc -Wall -O3 -fopenmp -o computepi-serial computepi-serial.c

computepi-omp: computepi-omp.c
	gcc -Wall -O3 -fopenmp -o computepi-omp computepi-omp.c

clean:
	/bin/rm computepi-serial computepi-omp