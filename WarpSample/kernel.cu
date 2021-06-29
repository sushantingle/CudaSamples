
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_warp_details()
{
	int gId = blockIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	int warpIdx = threadIdx.x / 32;
	int gbidx = blockIdx.y * gridDim.x + blockIdx.x;

	printf("GlobalIdx : %d, WarpIdx : %d, GlobalBlock : %d\n", gId, warpIdx, gbidx);

}

__global__ void non_divergent_func()
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = 0.0f;
	b = 0.0f;

	int warpId = gid / 32;
	if (warpId % 2 == 0)
	{
		a = 100.0f;
		b = 50.0f;
	}
	else
	{
		a = 200.0f;
		b = 200.0f;
	}
}

__global__ void divergent_func()
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = 0.0f;
	b = 0.0f;

	if (gid % 2 == 0)
	{
		a = 100.0f;
		b = 50.0f;
	}
	else
	{
		a = 200.0f;
		b = 200.0f;
	}
}

int main()
{
	//***** Warp Details Sample *****//

	//dim3 blocks(42);
	//dim3 grid(2, 2);

	//print_warp_details << <grid, blocks >> > ();
	//cudaDeviceSynchronize();
	
	//******************************//

	//***** Warp Divergence Sample *****//

	int size = 1 << 22;
	dim3 block_size(128);
	dim3 grid_size((size + block_size.x - 1) / block_size.x);

	non_divergent_func << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	divergent_func << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	//******************************//
	return 0;
}
