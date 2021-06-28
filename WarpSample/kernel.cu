
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

int main()
{
	dim3 blocks(42);
	dim3 grid(2, 2);

	print_warp_details << <grid, blocks >> > ();
	cudaDeviceSynchronize();

	return 0;
}
