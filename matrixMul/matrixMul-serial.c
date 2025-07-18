#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library

int main()
{
    const int M = 512, N = 1024, P = M;
    double *A = 0, *B = 0, *C = 0;
    double t0, dt;

    posix_memalign((void **) &A, 4096, M * N * sizeof(double));
    posix_memalign((void **) &B, 4096, N * P * sizeof(double));
    posix_memalign((void **) &C, 4096, M * P * sizeof(double));

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (double) (i + j);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < P; j++)
        {
            B[i * P + j] = (double) (i == j ? 1 : 0);
        }
    }

    t0 = omp_get_wtime();
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < P; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
    }
    dt = omp_get_wtime() - t0;

    printf("dt: %f\n", dt * 1e3);

    return 0;
}