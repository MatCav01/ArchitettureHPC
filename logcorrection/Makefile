all: logcorrection-serial logcorrection-omp

logcorrection-serial: logcorrection-serial.c
	gcc -Wall -O3 -fopenmp -o logcorrection-serial logcorrection-serial.c -lm

logcorrection-omp: logcorrection-omp.c
	gcc -Wall -O3 -fopenmp -o logcorrection-omp logcorrection-omp.c -lm

clean:
	/bin/rm logcorrection-serial logcorrection-omp