
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 1000000
#define ARRAY_SIZE 10

#define BLOCK_WIDTH 1000

void print_array(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d\n", arr[i]);
    }
}

__global__ void increment_atomic(int* g) {
    int i = blockDim.x * blockDim.x + threadIdx.x;

    i = i % ARRAY_SIZE;
    atomicAdd(&g[i], 1);
}

int main()
{
    GpuTimer timer;
    printf("Num of threads: %10d, Num of block: %10d, Number of array elements: %5d\n", NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    int h_array[ARRAY_SIZE]; // h stands for host (CPU)
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    int* d_array;
    cudaMalloc((void**)&d_array, ARRAY_BYTES);
    cudaMemset((void*)d_array, 0, ARRAY_BYTES);

    // launching the kernel
    timer.Start();
    increment_atomic<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >>>(d_array);
    timer.Stop();

    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time Elapsed: %g ms\n", timer.Elapsed());

    cudaFree(d_array);
    return 0;
}