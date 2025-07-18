all: derivative-serial derivative-omp

derivative-serial: derivative-serial.c
	gcc -Wall -O3 -fopenmp -o derivative-serial derivative-serial.c -lm

derivative-omp: derivative-omp.c
	gcc -Wall -O3 -fopenmp -o derivative-omp derivative-omp.c -lm

clean:
	/bin/rm derivative-serial derivative-omp