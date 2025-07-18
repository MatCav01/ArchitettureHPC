#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h> // sin library
#include <omp.h> // #pragma omp library
#define PI 3.14159265358979323846

inline double f(double x)
{
    return sin(x);
}

int main()
{
    int n;
    double a, b, h, dx, xi, t0, dt;
    double *Y;
    FILE *fp;

    a = 0.1;
    b = 2 * PI;
    n = 100;
    h = 1e-6;
    dx = (b - a) / n;

    if (posix_memalign((void **) &Y, 4096, n * sizeof(double)) != 0)
    {
        perror("ERROR: allocation of Y FAILED:");
        exit(-1);
    }

    // calculate derivative with central difference formula
    t0 = omp_get_wtime();
    for (int i = 0; i < n; i++)
    {
        xi = a + i * dx;
        Y[i] = (f(xi + h) - f(xi - h)) / (2 * h);
    }
    dt = omp_get_wtime() - t0;

    printf("n: %d  dt: %f msec\n", n, dt * 1e3);

    fp = fopen("results.out", "w+");
    for (int i = 0; i < n; i++)
    {
        fprintf(fp, "%f %f\n", a + i * dx, Y[i]);
    }
    fclose(fp);

    return 0;
}