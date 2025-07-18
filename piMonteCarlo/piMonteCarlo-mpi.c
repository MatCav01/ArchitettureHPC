// #define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h>
// #include <omp.h> // #pragma omp library
#include <mpi.h> // mpi library

typedef unsigned long long int ulli;

int main(int argc, char *argv[])
{
    double x, y, pi;
    int N, rank;
    ulli total_hits = 0, rank_hits = 0;
    ulli total_darts;
    ulli rank_darts = (ulli) 1e8;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &N);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // rank_darts = (total_darts % N == 0) ? (total_darts / N) : (total_darts / N + 1);
    total_darts = ((ulli) N) * rank_darts;

    srand(42 + rank);

    for (int i = 0; i < rank_darts; i++)
    {
        x = ((double) rand()) / RAND_MAX;
        y = ((double) rand()) / RAND_MAX;
        
        if (x * x + y * y <= 1)
        {
            rank_hits++;
        }
    }

    MPI_Reduce(&rank_hits, &total_hits, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        pi = 4.0 * (double) total_hits / (double) total_darts;

        printf("There were %lld hits in the circle out of %lld throws\n", total_hits, total_darts);
        printf("The estimated value of pi is: %.10f\n", pi);
        printf("The actual value is: %.10f\n", M_PI);
    }

    MPI_Finalize();

    return 0;
}