
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// unaligned global load
__global__ void unaligned_memory_access(float* a, float* b, float* c, int size, int offset)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int inputIdx = gid + offset;
    if (inputIdx < size)
    {
        c[gid] = a[inputIdx] + b[inputIdx];
    }
}

// unaligned global store
__global__ void unaligned_memory_store(float* a, float* b, float* c, int size, int offset)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int outputIdx = gid + offset;
    if (outputIdx < size)
    {
        c[outputIdx] = a[gid] + b[gid];
    }
}

int main(int argc, char** argv)
{
    int size = 1 << 22;
    int block_size = 128;
    int bytes = size * sizeof(float);
    int offset = 0;
    int kernel = 0;
    if (argc > 1)
        offset = atoi(argv[1]);

    if (argc > 2)
        kernel = atoi(argv[2]);

    float* h_A, *h_B;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);

    float* d_A, * d_B, *d_C;
    cudaMalloc((float**)&d_A, bytes);
    cudaMalloc((float**)&d_B, bytes);
    cudaMalloc((float**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid(size / block.x);

    switch (kernel)
    {
    case 1:
        unaligned_memory_access << <grid, block >> > (d_A, d_B, d_C, size, offset);
        break;
    case 2:
        unaligned_memory_store << <grid, block >> > (d_A, d_B, d_C, size, offset);
        break;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    cudaDeviceReset();
    return 0;
}
