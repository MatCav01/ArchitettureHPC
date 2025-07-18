#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#define MAXNUMTHREAD 16

int main()
{
    long n;
    double dx, xi, S, pi, t0, dt;

    n = 1e9;
    dx = 1.0 / (double) n;
    
    for (int t = 1; t <= MAXNUMTHREAD; t++)
    {
        S = 0.0;

        t0 = omp_get_wtime();

        #pragma omp parallel for num_threads(t) reduction(+ : S)
        for (int i = 1; i <= n; i++)
        {
            xi = (i - 0.5) * dx;
            S += 4.0 / (1.0 + xi * xi);
        }

        pi = dx * S; // dx è stato raccolto

        dt = omp_get_wtime() - t0;

        printf("t: %d  dt: %f sec  pi: %.10f\n", t, dt, pi);
    }

    return 0;
}