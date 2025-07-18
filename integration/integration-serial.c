#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h> // sin, cos library
#include <omp.h> // #pragma omp library

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
    int log10nmax;
    long n;
    double a, b, dx, xi, S, ve, er, t0, dt;

    a = 0.0;
    b = 10.0;
    log10nmax = 8;

    ve = F(b) - F(a);

    for (int i = 2; i < log10nmax; i++)
    {
        n = (long) pow(10, i);
        dx = (b - a) / n;
        S = 0.0;

        t0 = omp_get_wtime();
        for (long ii = 1; ii <= n; ii++)
        {
            xi = a + (double) (ii * dx);
            S += f(xi) * dx;
        }
        dt = omp_get_wtime() - t0;

        er = fabs(S - ve) / ve;

        printf("n: %ld  S: %f  ve: %f  err: %f  dt: %f msec\n", n, S, ve, er, dt * 1e3);
    }

    return 0;
}