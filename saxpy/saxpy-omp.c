// #define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#define NUM_THREADS 16

int main()
{
    float *X, *Y;
    float a = 17.17;
    long L = 1e8;
    double t0, dt;
    double sum = 0.0;

    posix_memalign((void **) &X, 4096, L * sizeof(float));
    posix_memalign((void **) &Y, 4096, L * sizeof(float));

    srand48(1999);
    for (int i = 0; i < L; i++)
    {
        X[i] = (float) drand48();
        Y[i] = (float) drand48();
    }

    t0 = omp_get_wtime();

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < L; i++)
    {
        Y[i] = a * X[i] + Y[i];
    }
    
    dt = omp_get_wtime() - t0;

    for (int i = 0; i < L; i++)
    {
        sum += Y[i];
    }

    free(X);
    free(Y);

    printf("sum: %0.2f  dt: %0.2f msec\n", sum, dt * 1e3);

    return 0;
}