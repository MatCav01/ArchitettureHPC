#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#include <cuda.h> // cuda library
#include <cuda_runtime.h>
#define TILEWIDTH 16

__global__ void matrixMul_kernel(int M, int N, int P, double *A, double *B, double *C);

int main()
{
    const int M = 512, N = 1024, P = M;
    double *A_h = 0, *B_h = 0, *C_h = 0;
    double *A_d, *B_d, *C_d;
    double t0, dt;

    dim3 dimBlock (TILEWIDTH, TILEWIDTH, 1);
    dim3 dimGrid (P / TILEWIDTH, M / TILEWIDTH, 1);

    posix_memalign((void **) &A_h, 4096, M * N * sizeof(double));
    posix_memalign((void **) &B_h, 4096, N * P * sizeof(double));
    posix_memalign((void **) &C_h, 4096, M * P * sizeof(double));

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A_h[i * N + j] = (double) (i + j);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < P; j++)
        {
            B_h[i * P + j] = (double) (i == j ? 1 : 0);
        }
    }

    cudaMalloc((void **) &A_d, M * N * sizeof(double));
    cudaMalloc((void **) &B_d, N * P * sizeof(double));
    cudaMalloc((void **) &C_d, M * P * sizeof(double));

    t0 = omp_get_wtime();
    
    cudaMemcpy(A_d, A_h, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * P * sizeof(double), cudaMemcpyHostToDevice);

    matrixMul_kernel <<<dimGrid, dimBlock>>> (M, N, P, A_d, B_d, C_d);

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, M * P * sizeof(double), cudaMemcpyDeviceToHost);

    dt = omp_get_wtime() - t0;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);

    printf("dt: %f\n", dt * 1e3);

    return 0;
}

__global__ void matrixMul_kernel(int M, int N, int P, double *A, double *B, double *C)
{
    __shared__ double As[TILEWIDTH][TILEWIDTH];
    __shared__ double Bs[TILEWIDTH][TILEWIDTH];
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < P && idx_y < M)
    {
        C[idx_y * P + idx_x] = 0;
        for (int i = 0; i < N / blockDim.x; i++)
        {
            As[threadIdx.y][threadIdx.x] = A[idx_y * N + (i * blockDim.x + threadIdx.x)];
            Bs[threadIdx.y][threadIdx.x] = B[(i * blockDim.y + threadIdx.y) + idx_x];

            __syncthreads();

            for (int j = 0; j < blockDim.x; j++)
            {
                C[idx_y * P + idx_x] += As[threadIdx.y][j] * Bs[j][threadIdx.x];
                
                __syncthreads();
            }
        }
    }
}