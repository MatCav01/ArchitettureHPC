// #define _POSIX_C_SOURCE 200809L
// #define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#include <mpi.h> // mpi library

double launch_saxpy(int mpi_rank, long mpi_L, float a, float *mpi_X, float *mpi_Y);

int main(int argc, char *argv[])
{
    float *X = 0, *Y = 0, *mpi_X = 0, *mpi_Y = 0;
    float a = 17.17;
    long L = 1e8, mpi_L;
    double dt, sum = 0.0;
    int err, mpi_size, mpi_rank;

    err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init error!\n");
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
    {
        posix_memalign((void **) &X, 4096, L * sizeof(float));
        posix_memalign((void **) &Y, 4096, L * sizeof(float));
    }

    mpi_L = L / mpi_size;
    printf("mpi_L: %ld\n", mpi_L);

    posix_memalign((void **) &mpi_X, 4096, mpi_L * sizeof(float));
    posix_memalign((void **) &mpi_Y, 4096, mpi_L * sizeof(float));

    if (mpi_rank == 0)
    {
        srand48(1999);
        for (int i = 0; i < L; i++)
        {
            X[i] = (float) drand48();
            Y[i] = (float) drand48();
        }

        double sum_pre = 0.0;
        for (int i = 0; i < L; i++)
        {
            sum_pre += Y[i];
        }
        printf("mpi_rank: %d  sum_pre: %e\n", mpi_rank, sum_pre);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // t0 = omp_get_wtime();
    
    err = MPI_Scatter(X, mpi_L, MPI_FLOAT, mpi_X, mpi_L, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scatter(X) error!");
        exit(-1);
    }
    err = MPI_Scatter(Y, mpi_L, MPI_FLOAT, mpi_Y, mpi_L, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scatter(Y) error!");
        exit(-1);
    }

    double sum_rank = 0.0;
    for (int i = 0; i < mpi_L; i++)
    {
        sum_rank += mpi_Y[i];
    }
    printf("mpi_rank: %d  sum_rank: %e\n", mpi_rank, sum_rank);

    dt = launch_saxpy(mpi_rank, mpi_L, a, mpi_X, mpi_Y);

    sum_rank = 0.0;
    for (int i = 0; i < mpi_L; i++)
    {
        sum_rank += mpi_Y[i];
    }
    printf("mpi_rank: %d  sum_rank: %e\n", mpi_rank, sum_rank);

    err = MPI_Gather(mpi_Y, mpi_L, MPI_FLOAT, Y, mpi_L, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Gather error!");
        exit(-1);
    }
    
    // dt = omp_get_wtime() - t0;

    if (mpi_rank == 0)
    {
        for (int i = 0; i < L; i++)
        {
            sum += Y[i];
        }

        printf("sum: %e  dt: %.2f msec\n", sum, dt * 1e3);

        free(X);
        free(Y);
    }

    free(mpi_X);
    free(mpi_Y);

    MPI_Barrier(MPI_COMM_WORLD);

    err = MPI_Finalize();
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Finalize error!\n");
        exit(-1);
    }

    return 0;
}