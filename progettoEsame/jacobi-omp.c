#define _POSIX_C_SOURCE 200809L
// #define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <string.h>
#include <omp.h> // #pragma omp library
// #include <mpi.h> // mpi library
#include <sys/param.h>
#include "common.h"
#define MAXNUMTHREADS 16

int main()
{
    int err;
    double *grid, *grid_new, *grid_tmp;
    double t0, dt, chk;

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

#if DUMP == 2
    FILE *myfile = fopen("data.dat", "w");
#endif

    for (int th = 1; th <= MAXNUMTHREADS; th++)
    {
        init(grid);

        memcpy(grid_new, grid, GX * GY * sizeof(double));

        t0 = omp_get_wtime();

        for(int iter = 1; iter <= MAXITER; iter++)
        {
            #pragma omp parallel for shared(grid) num_threads(th) schedule(static)
            for (int i = HY; i < GY - HY; i++)
            {
                // int th = omp_get_thread_num();
                // if (iter == 1)
                // {
                //     printf("Thread %d is doing row %d\n", th, i);
                // }
                for (int j = HY; j < GX - HX; j++)
                {
                    grid_new[i * GX + j] = (grid[i * GX + j] + grid[(i - 1) * GX + j] + grid[(i + 1) * GX + j] + grid[i * GX + j - 1] + grid[i * GX + j + 1]) / 5;
                }
            }

            grid_tmp = grid;
            grid = grid_new;
            grid_new = grid_tmp;
        }

        dt = omp_get_wtime() - t0;

        chk = checksum(grid);

        printf("[statistics] %dx%d  %d iter  th: %d  dt: %.3f msec  dt/iter: %.3f usec  GFLOPS: %.3f  checksum: %f\n",
            GLX, GLY, MAXITER, th, dt * 1e3, dt * 1e6 / (double)MAXITER, 5.0 * (double)MAXITER * (double)GLX * (double)GLY / (dt * 1e6), chk);
        
#if DUMP == 2
        fprintf(myfile, "th: %d  dt: %f\n", th, dt);
#endif

    }

#if DUMP == 2
    fclose(myfile);
#endif

    return 0;
}