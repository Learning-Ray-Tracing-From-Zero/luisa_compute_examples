#include <cstdio>
#include <cuda_runtime.h>


__global__ void init_array(int* arr) {
    if (blockDim.x * blockIdx.x + threadIdx.x == 0) {
        arr[0] = 3;
        arr[1] = 1;
        arr[2] = 7;
        arr[3] = 0;
        arr[4] = 4;
        arr[5] = 1;
        arr[6] = 6;
        arr[7] = 3;
    }
}

__global__ void block_prefix_sum(int* arr, int n) {
    int thread_x = threadIdx.x;
    for (int offset = 1; offset < n; offset *= 2) {
        int index = thread_x + offset;
        if (index < n) { arr[index] += arr[thread_x]; }
        __syncthreads();
    }
}


int main() {
    int n = 8;
    int *arr;
    cudaMallocManaged(&arr, n * sizeof(int));

    init_array<<<1, 1, 1>>>(arr);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) { printf("%d ", arr[i]); }
    printf("\n");

    block_prefix_sum<<<1, 8>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) { printf("%d ", arr[i]); }

    cudaFree(arr);
    return 0;
}
