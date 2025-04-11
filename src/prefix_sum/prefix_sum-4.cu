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
    extern __shared__ float shared_arr[];
    int thread_x = threadIdx.x;
    if (thread_x < n) { shared_arr[thread_x] = arr[thread_x]; }
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        int index = thread_x + offset;
        if (index < n) { shared_arr[index] += shared_arr[thread_x]; }
        __syncthreads();
    }
    if (thread_x < n) { arr[thread_x] = shared_arr[thread_x]; }
}


int main() {
    int n = 8;
    int *arr;
    cudaMallocManaged(&arr, n * sizeof(int));

    init_array<<<1, 1>>>(arr);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) { printf("%d ", arr[i]); }
    printf("\n");

    block_prefix_sum<<<1, 8, 8 * sizeof(int)>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) { printf("%d ", arr[i]); }

    cudaFree(arr);
    return 0;
}
