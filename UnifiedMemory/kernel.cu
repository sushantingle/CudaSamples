/*
	This is test run to allocate Unified Memory.
	Unified Memory creates pool of managed memory, where each allocation from this memory pool is accessible
	on both GPU and CPU with same address or pointer.
	You can do static and dynamic allocation in this managed pool.
	__device__ __managed__ float y;
	cudaMallocManaged(devicePtr, size, flags(unused));
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

__global__ void Unified_Memory_Test(float* a, float* b, float* c, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

int main()
{
	int size = 1 >> 25;
	int bytes = size * sizeof(float);

	dim3 block_size(128);
	dim3 grid(size / block_size.x);
	float* d_dataA, * d_dataB, * d_dataC;

	// allocate mapped host memory
	cudaMallocManaged((float**)&d_dataA, bytes);
	cudaMallocManaged((float**)&d_dataB, bytes);
	cudaMallocManaged((float**)&d_dataC, bytes);

	Unified_Memory_Test << <grid, block_size >> > (d_dataA, d_dataB, d_dataC, size);

	cudaDeviceSynchronize();

	cudaFree(d_dataA);
	cudaFree(d_dataB);
	cudaFree(d_dataC);

	cudaDeviceReset();

	return 0;
}
