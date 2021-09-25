
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void aKernel()
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int r1, r2, res_diff;
    __shared__ int arr[512];
    arr[idx] = idx;
    r1 = arr[idx];
    printf("A: Thread %5d, value %5d\n", idx, arr[idx]);

    if (idx < 511)
        arr[idx] = arr[idx + 1];
    r2 = arr[idx];
    res_diff = r2 - r1;
    printf("B: Thread %5d, value %5d, d=%5d\n", idx, arr[idx], res_diff);
}

int main()
{
    aKernel <<<1, 512>>> ();
    return 0;
}
