
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

__global__ void neighboured_pair_reduction(int* data, int* temp, int size)
{
    int tId = threadIdx.x;
    int gId = blockDim.x * blockIdx.x + threadIdx.x;

    if (gId > size)
        return;

    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        if (tId % (2 * offset) == 0)
        {
            data[gId] += data[gId + offset];
        }
        __syncthreads();
    }

    if (tId == 0)
    {
        temp[blockIdx.x] = data[gId];
    }
}


int Reduction_CPU(int* data, int size)
{
    int result = 0;
    for (int i = 0; i < size; ++i)
    {
        result += data[i];
    }
    return result;
}

void Initialize(int* data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = i;
    }
}

int main()
{
    int size = 1 << 27; // 128MB size
    int byte_size = size * sizeof(int);
    int block_size = 128;

    int* h_data;
    int* h_ref;

    h_data = (int*)malloc(byte_size);
    Initialize(h_data, size);

    int CPU_Result = Reduction_CPU(h_data, size);

    dim3 block(block_size);
    dim3 grid(size / block.x);

    printf("kernel Launch Parameters : Grid %d      Block.x %d \n", grid.x, block.x);
    int partial_sum_array_size = sizeof(int) * grid.x;
    h_ref = (int*)malloc(partial_sum_array_size);

    int* d_input, *d_temp;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_temp, partial_sum_array_size);

    cudaMemset(d_temp, 0, partial_sum_array_size);
    cudaMemcpy(d_input, h_data, byte_size, cudaMemcpyHostToDevice);
    neighboured_pair_reduction << <grid, block >> > (d_input, d_temp, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_temp, partial_sum_array_size, cudaMemcpyDeviceToHost);

    int gpu_result = 0;
    for (int i = 0; i < grid.x; ++i)
    {
        gpu_result += h_ref[i];
    }

    if (CPU_Result == gpu_result)
    {
        printf("Results are same on CPU and GPU.\n");
    }

    cudaFree(d_input);
    cudaFree(d_temp);
    free(h_data);
    free(h_ref);

    cudaDeviceReset();
    return 0;
}
