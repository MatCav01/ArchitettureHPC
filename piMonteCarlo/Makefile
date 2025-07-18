
all: piMonteCarlo-mpi piMonteCarlo-serial

piMonteCarlo-mpi: piMonteCarlo-mpi.c
	mpicc -Wall -O3 -I${MPIINC} -L${MPILIB} -lm -o piMonteCarlo-mpi piMonteCarlo-mpi.c

piMonteCarlo-serial: piMonteCarlo-serial.c
	gcc -Wall -O3 -lm -o piMonteCarlo-serial piMonteCarlo-serial.c

clean: 
	rm piMonteCarlo-serial piMonteCarlo-mpi
