// #define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#include <cuda.h> // cuda library
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 128

__global__ void saxpy(long L, float a, float *X, float *Y)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < L)
    {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main()
{
    float *X_h, *Y_h, *X_d, *Y_d;
    float a = 17.17;
    long L = 1e8;
    double t0, dt;
    double sum = 0.0;
    int n_blocks;

    if (L % THREADS_PER_BLOCK == 0)
    {
        n_blocks = L / THREADS_PER_BLOCK;
    }
    else
    {
        n_blocks = L / THREADS_PER_BLOCK + 1;
    }

    dim3 dimBlock (THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid (n_blocks, 1, 1);

    posix_memalign((void **) &X_h, 4096, L * sizeof(float));
    posix_memalign((void **) &Y_h, 4096, L * sizeof(float));
    
    cudaMalloc((void **) &X_d, L * sizeof(float));
    cudaMalloc((void **) &Y_d, L * sizeof(float));

    srand48(1999);
    for (int i = 0; i < L; i++)
    {
        X_h[i] = (float) drand48();
        Y_h[i] = (float) drand48();
    }

    t0 = omp_get_wtime();

    cudaMemcpy(X_d, X_h, L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, Y_h, L * sizeof(float), cudaMemcpyHostToDevice);

    saxpy <<< dimGrid, dimBlock >>> (L, a, X_d, Y_d);

    cudaDeviceSynchronize();

    cudaMemcpy(Y_h, Y_d, L * sizeof(float), cudaMemcpyDeviceToHost);

    dt = omp_get_wtime() - t0;

    cudaFree(X_d);
    cudaFree(Y_d);
    
    for (int i = 0; i < L; i++)
    {
        sum += Y_h[i];
    }

    free(X_h);
    free(Y_h);

    printf("sum: %0.2f  dt: %0.2f msec\n", sum, dt * 1e3);

    return 0;
}