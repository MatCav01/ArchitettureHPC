// #define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library

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

    #pragma omp target enter data map(alloc: X[0:L], Y[0:L])
    // #pragma omp target enter data map(alloc: Y[0:L])

    t0 = omp_get_wtime();

    #pragma omp target update to(X[0:L], Y[0:L])
    // #pragma omp target update to(Y[0:L])

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < L; i++)
    {
        Y[i] = a * X[i] + Y[i];
    }

    #pragma omp target update from(Y[0:L])

    dt = omp_get_wtime() - t0;

    #pragma omp target exit data map(release: X[0:L], Y[0:L])
    // #pragma omp target exit data map(release: Y[0:L])

    for (int i = 0; i < L; i++)
    {
        sum += Y[i];
    }

    free(X);
    free(Y);

    printf("sum: %0.2f  dt: %0.2f msec\n", sum, dt * 1e3);

    return 0;
}