#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void saxpy(long L, float a, float *X, float *Y)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < L)
    {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

extern "C" double launch_saxpy(int mpi_rank, long mpi_L, float a, float *mpi_X, float *mpi_Y)
{
    float *X_d, *Y_d;
    double t0, dt;
    int threads_per_block = (mpi_L < 128) ? mpi_L : 128;
    long n_blocks;

    if (mpi_L % threads_per_block == 0)
    {
        n_blocks = mpi_L / threads_per_block;
    }
    else
    {
        n_blocks = mpi_L / threads_per_block + 1;
    }

    dim3 dimBlock (threads_per_block, 1, 1);
    dim3 dimGrid (n_blocks, 1, 1);

    double sum_k = 0.0;
    for (int i = 0; i < mpi_L; i++)
    {
        sum_k += mpi_Y[i];
    }
    printf("mpi_rank: %d  sum_k: %e\n", mpi_rank, sum_k);

    cudaSetDevice(mpi_rank);
    
    cudaMalloc((void **) &X_d, mpi_L * sizeof(float));
    cudaMalloc((void **) &Y_d, mpi_L * sizeof(float));

    // printf("mpi_rank: %d\n", mpi_rank);

    t0 = omp_get_wtime();

    cudaMemcpy(X_d, mpi_X, mpi_L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, mpi_Y, mpi_L * sizeof(float), cudaMemcpyHostToDevice);

    // cudaSetDevice(mpi_rank);

    saxpy <<<dimGrid, dimBlock>>> (mpi_L, a, X_d, Y_d);

    cudaDeviceSynchronize();

    cudaMemcpy(mpi_Y, Y_d, mpi_L * sizeof(float), cudaMemcpyDeviceToHost);

    dt = omp_get_wtime() - t0;

    cudaFree(X_d);
    cudaFree(Y_d);

    sum_k = 0.0;
    for (int i = 0; i < mpi_L; i++)
    {
        sum_k += mpi_Y[i];
    }
    printf("mpi_rank: %d  sum_k: %e\n", mpi_rank, sum_k);

    return dt;
}