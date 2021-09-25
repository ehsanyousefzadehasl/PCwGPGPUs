
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void aKernel()
{
    int idx = threadIdx.x;
    int r1, r2, res_diff;
    __shared__ int arr[512];
    arr[idx] = idx;
    printf("A: Thread %5d, value %5d\n", idx, arr[idx]);
    __syncthreads();
    r1 = arr[idx];

    if (idx < 511) {
        int temp = arr[idx + 1];
        __syncthreads();
        arr[idx] = temp;
    }
    r2 = arr[idx];
    res_diff = r2 - r1;
    printf("B: Thread %5d, value %5d, diff=%5d\n", idx, arr[idx], res_diff);
}

int main()
{
    aKernel<<<1, 512>>> ();
    return 0;
}
