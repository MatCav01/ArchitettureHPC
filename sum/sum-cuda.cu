#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#define CPU_THREADS 8
#define THREADS_PER_BLOCK 512

__global__ void sum(double *a_d, double *b_d, double *c_d, int N);

int main()
{
    double *a_h, *b_h, *c_h, *r_h;
    double *a_d, *b_d, *c_d;
    int N = 1024, err = 0;

    dim3 dimBlock (THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid (N / THREADS_PER_BLOCK, 1, 1);

    a_h = (double *) malloc(N * sizeof(double));
    b_h = (double *) malloc(N * sizeof(double));
    c_h = (double *) malloc(N * sizeof(double));
    r_h = (double *) malloc(N * sizeof(double));

    cudaMalloc((void **) &a_d, N * sizeof(double));
    cudaMalloc((void **) &b_d, N * sizeof(double));
    cudaMalloc((void **) &c_d, N * sizeof(double));

    for(int i = 0; i < N; i++)
    {
        a_h[i] = (double) i;
        b_h[i] = (double) i;
    }

    cudaMemcpy(a_d, a_h, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N * sizeof(double), cudaMemcpyHostToDevice);

    sum <<< dimGrid, dimBlock >>> (a_d, b_d, c_d, N);

    cudaMemcpy(c_h, c_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    #pragma omp parallel for num_threads(CPU_THREADS)
    for (int i = 0; i < N; i++)
    {
        r_h[i] = a_h[i] + b_h[i];
    }

    for (int i = 0; i < N; i++)
    {
        if (c_h[i] != r_h[i])
        {
            err++;
        }
    }

    if (err == 0)
    {
        printf("SUM IS CORRECT!\n");
    }
    else
    {
        printf("THERE ARE %d SUM ERRORS!\n", err);
    }

    free(a_h);
    free(b_h);
    free(c_h);
    free(r_h);

    return 0;
}

__global__ void sum(double *a, double *b, double *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}