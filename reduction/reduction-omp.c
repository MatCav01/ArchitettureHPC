#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#define MAXNUMTHREADS 16

int main()
{
    const int N = 1e8;
    double *A = 0;
    double res, t0, dt;

    posix_memalign((void **) &A, 4096, N * sizeof(double));

    for (int i = 0; i < N; i++)
    {
        A[i] = i + 1;
    }

    for (int th = 1; th <= MAXNUMTHREADS; th++)
    {
        res = 0.0;
        t0 = omp_get_wtime();

        #pragma omp parallel for num_threads(th) reduction(+ : res)
        for (int i = 0; i < N; i++)
        {
            res += A[i];
        }
        
        dt = omp_get_wtime() - t0;

        printf("Th: %d  Result: %.2f  dt: %f\n", th, res, dt * 1e3);
    }

    free(A);

    return 0;
}