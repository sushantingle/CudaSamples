
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

// warp shuffle
template<unsigned int block_size>
__global__ void reduction_smem_warp_shuffle(int* data, int* temp, int size)
{
    int tid = threadIdx.x;
    int BLOCK_OFFSET = block_size * blockIdx.x;
    int* i_data = data + BLOCK_OFFSET;

    __shared__ int smem[block_size];

    smem[tid] = i_data[tid];
    __syncthreads();

    if (block_size >= 1024 && tid < 512)
    {
        smem[tid] += smem[tid + 512];
        __syncthreads();
    }

    if (block_size >= 512 && tid < 256)
    {
        smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (block_size >= 256 && tid < 128)
    {
        smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (block_size >= 128 && tid < 64)
    {
        smem[tid] += smem[tid + 64];
        __syncthreads();
    }

    if (block_size >= 64 && tid < 32)
    {
        smem[tid] += smem[tid + 32];
        __syncthreads();
    }

    int local_sum = smem[tid];
    if (tid < 32)
    {
        local_sum += __shfl_down_sync(0xFF, local_sum, 16);
        local_sum += __shfl_down_sync(0xFF, local_sum, 8);
        local_sum += __shfl_down_sync(0xFF, local_sum, 4);
        local_sum += __shfl_down_sync(0xFF, local_sum, 2);
        local_sum += __shfl_down_sync(0xFF, local_sum, 1);
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = local_sum;
    }
}

// Shared memory

template<unsigned int block_size>
__global__ void reduction_shared_memory(int* data, int* temp, int size)
{
    int tid = threadIdx.x;
    int BLOCK_OFFSET = block_size * blockIdx.x;
    int* i_data = data + BLOCK_OFFSET;

    __shared__ int smem[block_size];

    smem[tid] = i_data[tid];
    __syncthreads();

    if (block_size >= 1024 && tid < 512)
    {
        smem[tid] += smem[tid + 512];
        __syncthreads();
    }

    if (block_size >= 512 && tid < 256)
    {
        smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (block_size >= 256 && tid < 128)
    {
        smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (block_size >= 128 && tid < 64)
    {
        smem[tid] += smem[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int* vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}


// **********************************************************//
// Unrolling Technique:

// complete unrolling
// time : 9.25ms without template
// time : 8.87ms with template
template<unsigned int block_size>
__global__ void reduction_complete_unrolling(int* data, int* temp, int size)
{
    int tid = threadIdx.x;
    int BLOCK_OFFSET = block_size * blockIdx.x * 4;
    int index = BLOCK_OFFSET + tid;
    int* i_data = data + BLOCK_OFFSET;

    if ((index + block_size) < size)
    {
        int a1 = data[index];
        int a2 = data[index + block_size];
        int a3 = data[index + 2 * block_size];
        int a4 = data[index + 3 * block_size];
        data[index] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    if (block_size >= 1024 && tid < 512)
    {
        i_data[tid] += i_data[tid + 512];
        __syncthreads();
    }

    if (block_size >= 512 && tid < 256)
    {
        i_data[tid] += i_data[tid + 256];
        __syncthreads();
    }
    if (block_size >= 256 && tid < 128)
    {
        i_data[tid] += i_data[tid + 128];
        __syncthreads();
    }
    if (block_size >= 128 && tid < 64)
    {
        i_data[tid] += i_data[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int* vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

// warp unrolling
// time : 9.77ms
__global__ void reduction_warp_unrolling(int* data, int* temp, int size)
{
    int tid = threadIdx.x;

    int BLOCK_OFFSET = blockDim.x * blockIdx.x * 4;
    int index = BLOCK_OFFSET + tid;
    int* i_data = data + BLOCK_OFFSET;

    if ((index + blockDim.x) < size)
    {
        int a1 = data[index];
        int a2 = data[index + blockDim.x];
        int a3 = data[index + 2 * blockDim.x];
        int a4 = data[index + 3 * blockDim.x];
        data[index] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 64; offset /= 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int* vmem = i_data;
        vmem[tid] += i_data[tid + 32];
        vmem[tid] += i_data[tid + 16];
        vmem[tid] += i_data[tid + 8];
        vmem[tid] += i_data[tid + 4];
        vmem[tid] += i_data[tid + 2];
        vmem[tid] += i_data[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

// Manually adding 2 blocks.
// Elapsed Time : 32.94ms
__global__ void reduction_unrolling_block2(int* data, int* temp, int size)
{
    int tid = threadIdx.x;

    int BLOCK_OFFSET = blockDim.x * blockIdx.x * 2;
    int index = BLOCK_OFFSET + tid;
    int* i_data = data + BLOCK_OFFSET;

    if ((index + blockDim.x) < size)
    {
        data[index] += data[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

// Manually adding 4 blocks.
// Elapsed Time : 18.15ms
__global__ void reduction_unrolling_block4(int* data, int* temp, int size)
{
    int tid = threadIdx.x;

    int BLOCK_OFFSET = blockDim.x * blockIdx.x * 4;
    int index = BLOCK_OFFSET + tid;
    int* i_data = data + BLOCK_OFFSET;

    if ((index + blockDim.x) < size)
    {
        int a1 = data[index];
        int a2 = data[index + blockDim.x];
        int a3 = data[index + 2 * blockDim.x];
        int a4 = data[index + 3 * blockDim.x];
        data[index] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}
// **********************************************************//

// **********************************************************//
// Neighboured pair reduction has lot of divergence. So, to overcome the divergence, we  will be implementing two approaches.
// - Force summation of neighbouring approach
// - Interleaved pair approach

// Elapsed Time : 58.8ms
__global__ void neighboured_pair_improved(int* data, int* temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int* i_data = data + blockDim.x * blockIdx.x;

    if (gid > size)
        return;

    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        int index = 2 * offset * tid;
        if (index < blockDim.x)
        {
            i_data[index] += i_data[index + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = data[gid];
    }
}

// Elapsed Time : 61.1ms
__global__ void interleaved_pair_reduction(int* data, int* temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int* i_data = data + blockDim.x * blockIdx.x;

    if (gid > size)
        return;

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        i_data[tid] += i_data[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = data[gid];
    }
}
// **********************************************************//
// Elapsed Time : 78.89ms
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

    // neighboured_pair_reduction << <grid, block >> > (d_input, d_temp, size);
    //neighboured_pair_improved << <grid, block >> > (d_input, d_temp, size);
    // interleaved_pair_reduction << <grid, block >> > (d_input, d_temp, size);
    /*grid = (size / block.x) / 2;
    reduction_unrolling_block2 << <grid, block >> > (d_input, d_temp, size);*/
    /*grid = (size / block.x) / 4;
    reduction_unrolling_block4<< <grid, block >> > (d_input, d_temp, size);*/
    //grid = (size / block.x) / 4;
    //reduction_warp_unrolling << <grid, block >> > (d_input, d_temp, size);
    //grid = (size / block.x) / 4;
    //reduction_complete_unrolling<1024> << <grid, block >> > (d_input, d_temp, size);
    reduction_shared_memory<128> << <grid, block >> > (d_input, d_temp, size);
    //reduction_smem_warp_shuffle<128> << <grid, block >> > (d_input, d_temp, size);
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
