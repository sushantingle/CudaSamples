
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BDIMX 32
#define BDIMY 32

#define PAD 1

// dynamic shared memory with Padding.
// This would avoid bank conflicts while writing data in shared memory, since all column writing would spread across banks diagonally.
__global__ void smem_set_col_read_row_dynamic_pad(int* out)
{
	extern __shared__ int tile[];

	int rowIndex = threadIdx.y * (blockDim.x + PAD) + threadIdx.x;
	int colIndex = threadIdx.x * (blockDim.x + PAD) + threadIdx.y;

	tile[colIndex] = colIndex;
	__syncthreads();

	out[rowIndex] = tile[rowIndex];
}

__global__ void smem_set_col_read_row_dynamic(int* out)
{
	extern __shared__ int tile[];

	int rowIndex = threadIdx.y * blockDim.x + threadIdx.x;
	int colIndex = threadIdx.x * blockDim.x + threadIdx.y;

	tile[colIndex] = colIndex;
	__syncthreads();

	out[rowIndex] = tile[rowIndex];
}

// Set values in shared memory in row major order, so that there will not be any bank conflicts.
// access mode 64bit
//------------------------------------------------------------------------ 
//|   B1   |   B2   |   B3   | ....................................|   B31   |
//| 0  | 32| 1 | 33 | 2 | 34 |.....................................| 31 | 63 |
//------------------------------------------------------------------------
// Read shared memory in column major, so there will be bank conflicts.
__global__ void smem_set_row_read_col(int* out)
{
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int tile[BDIMX][BDIMY];

	tile[threadIdx.y][threadIdx.x] = idx;

	// wait till all threads in thread block finished setting value in shared memory
	__syncthreads();

	out[idx] = tile[threadIdx.x][threadIdx.y];
}

// Set values in shared memory in column major order, so there will be bank conflicts.
// Read from shared memory in row major order, so therer will be no bank conflicts. It will serve data in one transaction.
__global__ void smem_set_col_read_row(int* out)
{
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int tile[BDIMX][BDIMY];

	tile[threadIdx.x][threadIdx.y] = idx;

	// wait till all threads in thread block finished setting value in shared memory
	__syncthreads();

	out[idx] = tile[threadIdx.y][threadIdx.x];
}

// Read and write in shared memory in row major, so no conflicts.
__global__ void smem_set_row_read_row(int* out)
{
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int tile[BDIMX][BDIMY];

	tile[threadIdx.y][threadIdx.x] = idx;

	// wait till all threads in thread block finished setting value in shared memory
	__syncthreads();

	out[idx] = tile[threadIdx.y][threadIdx.x];
}

int main(int argc, char** argv)
{

	int smemconfig = 0;

	// set access mode to 32 bit or 64 bit.
	if (smemconfig == 0)
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemConfig::cudaSharedMemBankSizeEightByte);
	}
	else
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemConfig::cudaSharedMemBankSizeFourByte);
	}

	int* d_data;
	cudaMalloc((int**)&d_data, BDIMX * BDIMY * sizeof(int));

	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);

	smem_set_row_read_col << <grid, block >> > (d_data);
	cudaDeviceSynchronize();
	smem_set_col_read_row << <grid, block >> > (d_data);
	cudaDeviceSynchronize();
	smem_set_row_read_row << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	smem_set_col_read_row_dynamic << <grid, block, sizeof(int) * BDIMX * BDIMY >> > (d_data);
	cudaDeviceSynchronize();

	smem_set_col_read_row_dynamic_pad << <grid, block, sizeof(int)* (BDIMX + PAD) * BDIMY >> > (d_data);
	cudaDeviceSynchronize();

	cudaFree(d_data);
	cudaDeviceReset();
	return 0;
}
