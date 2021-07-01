
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// When we allocate host memory and do HostToDevice Transfer, Cuda copies host data from Pagable memory to Pinned Memory.
// Since pagable memory can't be transferred directly to GPU, Pinned memory needs to be used.
// Below test program is to show how can we allocate pinned memory.

int main()
{
    int size = 1 << 22;
    int bytes = size * sizeof(float);

    float* h_data;
    float* d_data;

    // Allocate Pinned memory
    cudaMallocHost((float**) &h_data, bytes);
    cudaMalloc((float**)&d_data, bytes);

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFreeHost(h_data);

    cudaDeviceReset();
    return 0;
}
