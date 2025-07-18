#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library

int main()
{
    long n;
    double dx, xi, S, pi, t0, dt;

    n = 1e9;
    dx = 1.0 / (double) n;
    S = 0.0;

    t0 = omp_get_wtime();

    for (int i = 1; i <= n; i++)
    {
        xi = (i - 0.5) * dx;
        S += 4.0 / (1.0 + xi * xi);
    }

    pi = dx * S; // dx è stato raccolto

    dt = omp_get_wtime() - t0;

    printf("dt: %f sec  pi: %.10f\n", dt, pi);

    return 0;
}