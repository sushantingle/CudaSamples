
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Wrap Id
__global__ void Print_Warp_Details()
{
    int gIdx = blockIdx.y * gridDim.x * blockDim.x +
        blockDim.x * blockIdx.x +
        threadIdx.x;
    int warpIdx = threadIdx.x / 32;
    int gbidx = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid=%d, block Idx = %d, block Idy = %d, gid = %d, warp id = %d, gbid = %d\n", threadIdx.x, blockIdx.x, blockIdx.y, gIdx, warpIdx, gbidx);
}


// Wrap Divergence

int main()
{
    dim3 block_size(2);
    dim3 grid_size(2, 2);

    Print_Warp_Details << <grid_size, block_size>> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
        
    return EXIT_SUCCESS;
}

