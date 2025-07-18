// #define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h>
// #include <omp.h> // openmp library
// #include <mpi.h> // mpi library

typedef unsigned long long int ulli;

int main()
{
    double x, y;
    ulli hits = 0;
    ulli n_darts = (ulli) 1e8;

    srand(42);

    for(ulli i = 0; i < n_darts; i++)
    {
        x = ((double) rand()) / RAND_MAX;
        y = ((double) rand()) / RAND_MAX;

        if (x * x + y * y <= 1)
        {
            hits++;
        }
    }

    printf("There were %lld hits in the circle\n", hits);
    printf("The estimated value of pi is: %.10f\n", 4.0 * (double) hits / (double) n_darts);
    printf("The actual value is: %.10f\n", M_PI);

    return 0;
}