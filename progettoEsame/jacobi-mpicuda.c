#define _POSIX_C_SOURCE 200809L
// #define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <string.h>
#include <omp.h> // #pragma omp library
#include <mpi.h> // mpi library
#include "common.h"
#define MPIROOT 0

void launch_jacobi(int mpi_rank, double mpi_GY, double *mpi_grid);

int main(int argc, char *argv[])
{
    int err, mpi_size, mpi_rank, mpi_GLY, mpi_GY;
    double *grid = 0, *mpi_grid = 0;//, *mpi_grid_new = 0, *mpi_grid_tmp;
    double t0, dt, dt_max, chk;
    MPI_Status mpi_status;
#if DUMP == 1
    char myfile[32];
#endif

    err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init error!\n");
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == MPIROOT)
    {
        posix_memalign((void **) &grid, 4096, GX * GY * sizeof(double));
    }

    // mpi_GLY = (GLY % mpi_size == 0) ? (GLY / mpi_size) : (GLY / mpi_size + 1);
    mpi_GLY = GLY / mpi_size;
    mpi_GY = HY + mpi_GLY + HY;

    posix_memalign((void **) &mpi_grid, 4096, GX * mpi_GY * sizeof(double));
    // posix_memalign((void **) &mpi_grid_new, 4096, GX * mpi_GY * sizeof(double));

    if (mpi_rank == MPIROOT)
    {
        init(grid);

#if DUMP == 1
        sprintf(myfile, "video/grid-%07d", 0);
        dump(grid, myfile);
#endif

    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(grid + GX * HY, GX * mpi_GLY, MPI_DOUBLE, mpi_grid + GX * HY, GX * mpi_GLY, MPI_DOUBLE, MPIROOT, MPI_COMM_WORLD);
    
    // init edges
    if (mpi_rank == MPIROOT)
    {
        memcpy(mpi_grid, grid, GX * HY * sizeof(double));
        memcpy(mpi_grid + GX * (HY + mpi_GLY), grid + GX * (HY + mpi_GLY), GX * HY * sizeof(double));

        for (int rank = 1; rank < mpi_size; rank++)
        {
            MPI_Send(grid + GX * (HY + rank * mpi_GLY - HY), GX * HY, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
            MPI_Send(grid + GX * (HY + (rank + 1) * mpi_GLY), GX * HY, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(mpi_grid, GX * HY, MPI_DOUBLE, MPIROOT, mpi_rank, MPI_COMM_WORLD, &mpi_status);
        MPI_Recv(mpi_grid + GX * (HY + mpi_GLY), GX * HY, MPI_DOUBLE, MPIROOT, mpi_rank, MPI_COMM_WORLD, &mpi_status);
    }

    // memcpy(mpi_grid_new, mpi_grid, GX * mpi_GY * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);

    t0 = omp_get_wtime();

    for(int iter = 1; iter <= MAXITER; iter++)
    {
        launch_jacobi(mpi_rank, mpi_GY, mpi_grid);

        MPI_Barrier(MPI_COMM_WORLD);

        // edges update
        if (mpi_rank % 2 == 0)
        {
            if (mpi_rank > MPIROOT)
            {
                MPI_Send(mpi_grid + GX * HY, GX * HY, MPI_DOUBLE, mpi_rank - 1, mpi_rank, MPI_COMM_WORLD);
            }

            MPI_Send(mpi_grid + GX * mpi_GLY, GX * HY, MPI_DOUBLE, mpi_rank + 1, mpi_rank, MPI_COMM_WORLD);

            if (mpi_rank > MPIROOT)
            {
                MPI_Recv(mpi_grid, GX * HY, MPI_DOUBLE, mpi_rank - 1, mpi_rank - 1, MPI_COMM_WORLD, &mpi_status);
            }

            MPI_Recv(mpi_grid + GX * (HY + mpi_GLY), GX * HY, MPI_DOUBLE, mpi_rank + 1, mpi_rank + 1, MPI_COMM_WORLD, &mpi_status);
        }
        else
        {
            if (mpi_rank < mpi_size - 1)
            {
                MPI_Recv(mpi_grid + GX * (HY + mpi_GLY), GX * HY, MPI_DOUBLE, mpi_rank + 1, mpi_rank + 1, MPI_COMM_WORLD, &mpi_status);
            }

            MPI_Recv(mpi_grid, GX * HY, MPI_DOUBLE, mpi_rank - 1, mpi_rank - 1, MPI_COMM_WORLD, &mpi_status);

            if (mpi_rank < mpi_size - 1)
            {
                MPI_Send(mpi_grid + GX * mpi_GLY, GX * HY, MPI_DOUBLE, mpi_rank + 1, mpi_rank, MPI_COMM_WORLD);
            }

            MPI_Send(mpi_grid + GX * HY, GX * HY, MPI_DOUBLE, mpi_rank - 1, mpi_rank, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // mpi_grid_tmp = mpi_grid;
        // mpi_grid = mpi_grid_new;
        // mpi_grid_new = mpi_grid_tmp;
    }

    dt = omp_get_wtime() - t0;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(mpi_grid + GX * HY, GX * mpi_GLY, MPI_DOUBLE, grid + GX * HY, GX * mpi_GLY, MPI_DOUBLE, MPIROOT, MPI_COMM_WORLD);

    MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPIROOT, MPI_COMM_WORLD);

    if (mpi_rank == MPIROOT)
    {

#if DUMP == 1
    sprintf(myfile, "video/grid-%07d", MAXITER);
    dump(grid, myfile);
#endif

        chk = checksum(grid);

        printf("[statistics] %dx%d  %d iter  dt: %.3f msec  dt/iter: %.3f usec  GFLOPS: %.3f  checksum: %f\n",
            GLX, GLY, MAXITER, dt_max * 1e3, dt_max * 1e6 / (double)MAXITER, 5.0 * (double)MAXITER * (double)GLX * (double)GLY / (dt_max * 1e9), chk);
        
        free(grid);
    }

    free(mpi_grid);
    // free(mpi_grid_new);

    MPI_Barrier(MPI_COMM_WORLD);

    err = MPI_Finalize();
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Finalize error!\n");
        exit(-1);
    }

    return 0;
}