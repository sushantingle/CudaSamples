/**
This is test run to allocate zero copy memory.
Zero copy memory is memory mapped in device address space. This could be used to improve PCIe transfer.
We don't have to do explicit data transfer between host and device.

If device memory is frequently read/write, then it could be huge impact on performance,
since each operation needs to transferred to host on PCIe bus.
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

__global__ void Zero_Copy_Add(float* a, float* b, float* c, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

int main()
{
	int deviceId = 0;
	cudaSetDevice(deviceId);

	// Get Device Properties
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	if (!deviceProp.canMapHostMemory) // check if mapping is supported or not.
	{
		printf("Device %d does not support mapping CPU host memory.\n");
		return 0;
	}

	int size = 1 >> 25;
	int bytes = size * sizeof(float);

	dim3 block_size(128);
	dim3 grid(size / block_size.x);
	float* h_dataA, * h_dataB, * h_dataC;

	h_dataC = (float*)malloc(bytes);
	
	// allocate mapped host memory
	cudaHostAlloc((float**)&h_dataA, bytes, cudaHostAllocMapped);
	cudaHostAlloc((float**)&h_dataB, bytes, cudaHostAllocMapped);

	float* d_dataA, *d_dataB, *d_dataC;
	// Get Device pointer to mapped memory
	cudaHostGetDevicePointer((float**)&d_dataA, h_dataA, 0);
	cudaHostGetDevicePointer((float**)&d_dataB, h_dataB, 0);

	cudaMalloc((float**)&d_dataC, bytes);

	Zero_Copy_Add << <grid, block_size >> > (d_dataA, d_dataB, d_dataC, size);

	cudaDeviceSynchronize();
	
	cudaMemcpy(h_dataC, d_dataC, bytes, cudaMemcpyDeviceToHost);

	cudaFreeHost(h_dataA);
	cudaFreeHost(h_dataB);
	cudaFree(d_dataC);
	free(h_dataC);

	cudaDeviceReset();

	return 0;

}
