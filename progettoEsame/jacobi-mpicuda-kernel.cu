#define _POSIX_C_SOURCE 200809L
// #define _XOPEN_SOURCE 700
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <string.h>
#include <omp.h> // #pragma omp library
// #include <mpi.h> // mpi library
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#define TILEWIDTH 16

__global__ void jacobi_kernel(double mpi_GY, double *grid, double *grid_new);

extern "C" void launch_jacobi(int mpi_rank, double mpi_GY, double *mpi_grid)
{
    cudaError_t cuda_err;
    double *grid_d, *grid_new_d;
    // double t0, dt;

    int nblocksX = (GLX % TILEWIDTH == 0) ? (GLX / TILEWIDTH) : (GLX / TILEWIDTH + 1);
    int nblocksY = (GLY % TILEWIDTH == 0) ? (GLY / TILEWIDTH) : (GLY / TILEWIDTH + 1);

    dim3 dimBlock(TILEWIDTH, TILEWIDTH, 1);
    dim3 dimGrid(nblocksX, nblocksY, 1);

    cudaSetDevice(mpi_rank);

    cuda_err = cudaMalloc((void **) &grid_d, GX * mpi_GY * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "grid_d cudaMalloc error!\n");
        exit(-1);
    }
    cuda_err = cudaMalloc((void **) &grid_new_d, GX * mpi_GY * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "grid_new_d cudaMalloc error!\n");
        exit(-1);
    }

    // t0 = omp_get_wtime();

    cuda_err = cudaMemcpy(grid_d, mpi_grid, GX * mpi_GY * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "grid_d cudaMemcpy HostToDevice error!\n");
        exit(-1);
    }

    cuda_err = cudaMemcpy(grid_new_d, grid_d, GX * mpi_GY * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "grid_new_d cudaMemcpy DeviceToDevice error!\n");
        exit(-1);
    }

    // for(int iter = 1; iter <= MAXITER; iter++)
    // {
    // grid_new_d <-- grid_d
    jacobi_kernel <<<dimGrid, dimBlock>>> (mpi_GY, grid_d, grid_new_d);

    cudaDeviceSynchronize();

        // iter++;

        // // grid_d <-- grid_new_d
        // jacobi_kernel <<<dimGrid, dimBlock>>> (grid_new_d, grid_d);

        // cudaDeviceSynchronize();

    cuda_err = cudaMemcpy(mpi_grid, grid_new_d, GX * mpi_GY * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "grid_d cudaMemcpy DeviceToHost error!\n");
        exit(-1);
    }
    // }

    // dt = omp_get_wtime() - t0;

    cudaFree(grid_d);
    cudaFree(grid_new_d);

    // return dt;
}

__global__ void jacobi_kernel(double mpi_GY, double *grid, double *grid_new)
{
    // __shared__ double gridS[TILEWIDTH + 2][TILEWIDTH + 2];

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x + HX; // idx_x pixel
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y + HY; // idx_y pixel

    if (idx_x < GX - HX && idx_y < mpi_GY - HY)
    {
        // int x = threadIdx.x + 1;
        // int y = threadIdx.y + 1;
        
        // if (y == 1)
        // {
        //     gridS[y - 1][x] = grid[(idx_y - 1) * GX + idx_x];
        // }
        // else if (y == blockDim.y)
        // {
        //     gridS[y + 1][x] = grid[(idx_y + 1) * GX + idx_x];
        // }
        // if (x == 1)
        // {
        //     gridS[y][x - 1] = grid[idx_y * GX + idx_x - 1];
        // }
        // else if (x == blockDim.x)
        // {
        //     gridS[y][x + 1] = grid[idx_y * GX + idx_x + 1];
        // }
        // gridS[y][x] = grid[idx_y * GX + idx_x];
        
        // __syncthreads();

        // grid_new[idx_y * GX + idx_x] = (gridS[y][x] + gridS[y - 1][x] + gridS[y + 1][x] + gridS[y][x - 1] + gridS[y][x + 1]) / 5;

        // __syncthreads();

        grid_new[idx_y * GX + idx_x] = (grid[idx_y * GX + idx_x] + grid[(idx_y - 1) * GX + idx_x] + grid[(idx_y + 1) * GX + idx_x] + grid[idx_y * GX + idx_x - 1] + grid[idx_y * GX + idx_x + 1]) / 5;
    }
}