#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#include <cuda.h> // cuda library
#include <cuda_runtime.h>
#define TILEWIDTH 16

__global__ void reduction_kernel(int N, double *A, double *res);

int main()
{
    const int N = 1e8;
    double *A_h = 0, res_h;
    double *A_d, *res_d;
    double t0, dt;
    dim3 dimBlock (TILEWIDTH, 1, 1);
    dim3 dimGrid (N / TILEWIDTH, 1, 1);

    posix_memalign((void **) &A_h, 4096, N * sizeof(double));

    for (int i = 0; i < N; i++)
    {
        A_h[i] = i + 1;
    }

    cudaMalloc((void **) &A_d, N * sizeof(double));
    cudaMalloc((void **) &res_d, N / TILEWIDTH * sizeof(double));

    t0 = omp_get_wtime();

    cudaMemcpy(A_d, A_h, N * sizeof(double), cudaMemcpyHostToDevice);

    reduction_kernel <<<dimGrid, dimBlock>>> (N, A_d, res_d);

    cudaDeviceSynchronize();

    cudaMemcpy(&res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);

    dt = omp_get_wtime() - t0;

    cudaFree(A_d);
    cudaFree(res_d);

    free(A_h);

    printf("Result: %.2f  dt: %f\n", res_h, dt * 1e3);

    return 0;
}

__global__ void reduction_kernel(int N, double *A, double *res)
{
    __shared__ double As[TILEWIDTH];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    As[threadIdx.x] = (idx < N) ? A[idx] : 0.0;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            As[threadIdx.x] += As[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        res[blockIdx.x] = As[0];
    }
}