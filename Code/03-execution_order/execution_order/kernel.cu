
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
    printf("I am the thread of block %d\n", blockIdx.x);
}

int main()
{
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
    cudaDeviceSynchronize();

    printf("This is the END!");
    return 0;
}
