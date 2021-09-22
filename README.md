# Parallel Computing with GPGPUs
In this repository, I summarized what I reviewed and learned from parallel comptuing with the help of GPGPUs. If you are starting to learn about parallel computing and GPGPUs, "Intro to Parallel Computing", which nowadays is a little out-of-date, but still helpful, by Prof. John Ownes from UC Davis would be a real eye-opener.

## Introduction
In this very first section, you will get familiar with a little story 
### Narrative History!
In this very short and concise introduction, I just want to summarize the reasons why we moved toward parallelism, and why GPGPUs got famous. The story starts 1948 (I can hardly remember, I do not think you would be able to remember anything) when Von Neumann architecture became the mainstream. Following this architecture, computer architectures started desigining and building processors that were fetching data from memories, processing, then writing back to memories. They encountered several challenges, but tried to find solutions for them. One of the challenges was the processor-memory performance gap. Computer architects for addressing this challenge focused on architectural techniques like caching, pre-fetching, multi-threding, PIM. The other one was memory wall when computer architects were not able to improve the performance simply by increasing the working frequency of chips due to the dennard scaling breakdown. So, they steered the computer architecture trend toward Parallelism. This time instead of complex large processor cores, they were desigining more simple processors working together. This architecture increased performance, and power efficiency by providing more operations per watt. Indeed, they focused on throughput (on large cores their focus was on latency) in these architectures. The only of parallel systems was the programmability hardship which was posed on programmers. Most of the time, it is challenging for a programmer who is used to develop serial program, to switch to a new thinking paradigm and develop parallel programs!

## Compute Unified Device Architecture (CUDA)
It is a parallel computing platform and API created by NVIDIA allowing developers to use a CUDA-enabled GPU for general purpose processing. This approach is termed as GPGPU (General Purpose GPU). The CUDA platform is a software layer providing a direct access to the GPU's virtual instruction set (PTX) and parallel computational elements, for the execution of kernels which also are called computer kernels. This platform is designed to work with programming languages like C, C++, Fortran. As a result, programming with CUDA is much easier than prior APIs, like Direct3D and OpenGL, to it.

This repository's goal is working with this platform. For working with this platform on Windows, you must install Microsoft Visual Studio Code alongside Nvidia's CUDAtoolkit. Nvidia Nsight is an application development environment which brings GPU computing into Microsoft Visual Studioallows you to build and debug integrated GPU kernels and native CPU code as well as inspect the state of the CPU, GPU, and memory. You do it because you need a c/c++ compiler beside NVCC. But, on a Linux OS due to the built-in compilers, you don't need to Microsoft VS.

When we develop a cuda program (extension .cu), it is consisted of two parts: (1) one part runs on the processor (CPU) which is usually called host processor, (2) the other one runs on a GPU, which is usually called device. The following figure shows how a cuda program runs on a system consisted of a CPU and a GPU (called Heterogeneous System). When we write a program in C, C++, Python or other programming language, it executes only on the CPU. However, CUDA platform makes us to write one code that will be executed on both CPU and GPU.

![CUDA program](Images/CUDA_program.jpg)

A CUDA program, which views GPU as a co-processor, CPU or host is in charge of doing the data movements between CPU and GPU memories with Memcpy CUDA instruction (1, 2). It also allocates a memory part on GPU's memory with Memalloc CUDA instruction (3). Then, it launches kernels on GPU to be executed by GPU with three arrow syntax, which we will see together soon.

Note that:
1. GPU can only respond to CPU requests for sending and receiving data from/ to CPU. Also, it computes a kernel launched by CPU.
2. GPU cannot initiate any data sending/ receiving request.
3. GPU cannot compute a kernel launched by itself or it cannot launch a kernel. In other words, from a kernel another kernel cannot be launched.
4. CPU launches kernels on GPU in the order you write in your code.

### A Typical GPU Program
In a CUDA program, the following list happens:
1. CPU allocates memory on GPU (cudaMalloc)
2. CPU copies input data from its memory to GPU's memory (cudaMemcpy)
3. CPU launches kernel(s) on GPU to process the input data (copied in the previous step)
4. CPU copies results back to its memory from GPU's memory (cudaMemcpy)

The big idea in writing GPU programs is that kernels look like serial programs as if it will run on thread. The GPU will run that program on many threads.

Also, note that in GPU computing, throughput is what will give you efficiency and performance (or GPUs optimize for throughput). GPUs are efficient when your program is highly parallel. GPUs are masters of launching a large number of threads efficiently. The data movement part should be considered unless you will get totally nothing while burning more energy uselessly.

### A Simple Program
#### With a CPU
Let's assume that we want to compute the square of each element of an array. First, let's see the serial code of it which runs only on a CPU.

```c
length_of_array = 1024;
for(int i = 0; i < length_of_array; i++) {
    out[i] = in[i] * in[i];
}
```

In this program, there is no explicit parallelism, so it is executed by a single thread. As a result, 1024 multiplications are required for the task completion. let's assume that each multiplication takes 2 nanoseconds, so:

```
Execution time = 1024 * 2 ns = 2048 ns
```

#### With a GPU
Let's do this computation with the help of a GPU. In this scenario, CPU will move data between its memory and GPU memory and launch kernel, which impose timing overheads. GPU will launch 1024 threads (degree of parallelism is 1024). Each thread will multiply each array element to itself and put it in the result array. So, Only one multiply time by GPU takes to finish the task because all of the threads will begin their execution simultaneously.

```c
#include <stdio.h>

__global__ void square(float * d_out, float * d_in) {
    int idx = threadIdx.x; // threadIdx is a cuda built-in variable
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main() {
    const int ARRAY_SIZE = 1024;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float * d_in;
    float * d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```

If each multiplication by GPU takes 10 nanoseconds and there are some overheads imposed by data transfer and kernel launching.

```
Execution Time = 10 ns + 2 * DT Overhead + L Overhead 
```

As it is evident, if the data transfer overhead is negligible compared to the parallelism, we can gain a lot from GPU computing.

We will learn more about the kernel, its configuration, especially about thread blocks, number of threads in a block, and cuda built-in variables for addressing threads on GPU. For now, it is enough to know that in kernel launching the first argument shows the number of thread blocks, and the second one shows the number of threads per thread block. dim3 can be used to describe the dimension of blocks and threads. The general kernel launch looks like the following snippet:

```c
kernel_name<<<dim3(Dx, Dy, Dz), dim3(Tx, Ty, Tz), shmem>>>(argment list);
```

shmem shows the shared memory per block in bytes.

The following list briefly introduces you the built-in variables of CUDA for addressing threads and blocks:

1. threadIdx: thread within block - threadIdx.x, threadIdx.y
2. blockDim: size of a block (# of threads in it)
3. blockIdx: block within grid
4. gridDim: size of grid (# of blocks in it)

In the following example: our kernel is consisted of 8 blocks, each containing 64 threads. Totally, the kernel consists of 64 * 8 = 512

```c
kernel_name<<<dim3(2, 2, 2), dim3(4, 4, 4)>>>(argment list);
```
Note: for calculating the data transfer time overhead between CPU and GPU, [NVIDIA Nsight Systems tool](https://developer.nvidia.com/nsight-systems) can be used.

