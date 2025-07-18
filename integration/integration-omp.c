#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h> // sin, cos library
#include <omp.h> // #pragma omp library
#define MAXNUMTHREAD 16

inline double f(double x)
{
    return sin(x);
}

inline double F(double x)
{
    return -cos(x);
}

int main()
{
    long n;
    double a, b, dx, xi, S, ve, er, t0, dt;

    a = 0.0;
    b = 10.0;
    n = 1e8;
    dx = (b - a) / n;

    ve = F(b) - F(a);

    for (int t = 1; t <= MAXNUMTHREAD; t++)
    {
        S = 0.0;

        t0 = omp_get_wtime();

        #pragma omp parallel for num_threads(t) reduction(+ : S)
        for (long ii = 1; ii <= n; ii++)
        {
            xi = a + (double) (ii * dx);
            S += f(xi) * dx;
        }
        
        dt = omp_get_wtime() - t0;

        er = fabs(S - ve) / ve;

        printf("t: %d  S: %f  ve: %f  err: %f  dt: %f msec\n", t, S, ve, er, dt * 1e3);
    }

    return 0;
}