#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


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

__global__ void block_prefix_sum(int* in_arr, int n) {
    int thread_x = threadIdx.x;

    // load input data into shared memory
    extern __shared__ int shared_arr[];
    int x = thread_x;
    int y = thread_x + (n / 2);
    int bank_offset_a = CONFLICT_FREE_OFFSET(x);
    int bank_offset_b = CONFLICT_FREE_OFFSET(y);
    shared_arr[x + bank_offset_a] = in_arr[x];
    shared_arr[y + bank_offset_b] = in_arr[y];

    // up sweep
    int offset = 1;
    for (int d = n >> 1; d >= 1; d >>= 1) {
        if (thread_x < d) {
            int ai = offset * (2 * thread_x + 1) - 1;
            int bi = offset * (2 * thread_x + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            shared_arr[bi] += shared_arr[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // down sweep
    if (thread_x == 0) { // clear the last element
        shared_arr[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        if (thread_x < d) {
            int ai = offset * (2 * thread_x + 1) - 1;
            int bi = offset * (2 * thread_x + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int temp = shared_arr[ai];
            shared_arr[ai] = shared_arr[bi];
            shared_arr[bi] += temp;
        }
        __syncthreads();
    }

    // write results to device memory
    in_arr[x] = shared_arr[x + bank_offset_a];
    in_arr[y] = shared_arr[y + bank_offset_b];
}


int main() {
    int n = 8;
    int *arr;
    cudaMallocManaged(&arr, n * sizeof(int));

    init_array<<<1, 1>>>(arr);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; ++i) { printf("%d ", arr[i]); }
    printf("\n");
    int last = arr[n - 1];

    block_prefix_sum<<<1, 4, n * sizeof(int)>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 1; i < n; ++i) { arr[i - 1] = arr[i]; }
    arr[n - 1] = arr[n - 2] + last;
    for (int i = 0; i < n; ++i) { printf("%d ", arr[i]); }

    cudaFree(arr);
    return 0;
}
