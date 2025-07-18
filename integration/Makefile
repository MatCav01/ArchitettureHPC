all: integration-serial integration-omp

integration-serial: integration-serial.c
	gcc -Wall -O3 -fopenmp -o integration-serial integration-serial.c -lm

integration-omp: integration-omp.c
	gcc -Wall -O3 -fopenmp -o integration-omp integration-omp.c -lm

clean:
	/bin/rm integration-serial integration-omp