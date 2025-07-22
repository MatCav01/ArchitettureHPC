#define _POSIX_C_SOURCE 200809L
// #define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <string.h>
#include <omp.h> // #pragma omp library
// #include <mpi.h> // mpi library
#include "common.h"

int main()
{
    int err;
    double *grid, *grid_new, *grid_tmp;
    double t0, dt;
#if DUMP == 1
    char myfile[32];
#endif

    err = posix_memalign((void **) &grid, 4096, GX * GY * sizeof(double));
    if (err != 0)
    {
        fprintf(stderr, "grid posix_memaling error!");
        exit(-1);
    }
    err = posix_memalign((void **) &grid_new, 4096, GX * GY * sizeof(double));
    if (err != 0)
    {
        fprintf(stderr, "grid_new posix_memaling error!");
        exit(-1);
    }

    init(grid);

    memcpy(grid_new, grid, GX * GY * sizeof(double));

#if DUMP == 1
    sprintf(myfile, "video/grid-%07d", 0);
    dump(grid, myfile);
#endif

    t0 = omp_get_wtime();

    for(int iter = 1; iter <= MAXITER; iter++)
    {
        for (int i = HY; i < GY - HY; i++)
        {
            for (int j = HY; j < GX - HX; j++)
            {
                grid_new[i * GX + j] = (grid[i * GX + j] + grid[(i - 1) * GX + j] + grid[(i + 1) * GX + j] + grid[i * GX + j - 1] + grid[i * GX + j + 1]) / 5;
            }
        }

#if DUMP == 1
        if (iter % DUMPSTEP == 0)
        {
            sprintf(myfile, "video/grid-%07d", iter);
            dump(grid_new, myfile);
        }
#endif

        grid_tmp = grid;
        grid = grid_new;
        grid_new = grid_tmp;
    }

    dt = omp_get_wtime() - t0;

#if DUMP == 1
    sprintf(myfile, "video/grid-%07d", MAXITER);
    dump(grid, myfile);
#endif

    printf("[statistics] %dx%d  %d iter  dt: %.3f msec  dt/iter: %.3f usec  GFLOPS: %.3f\n",
        GLX, GLY, MAXITER, dt, dt * 1e3 / (double)MAXITER, 5.0 * (double)MAXITER * (double)GLX * (double)GLY / (dt * 1e6));

    return 0;
}