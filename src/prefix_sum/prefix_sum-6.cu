// Balanced Trees
// To build a balanced binary tree on the input data and
//   sweep it to and from the root to compute the prefix sum
// A binary tree with `n` leaves has `d = log2n` levels,
//   and each level `d` has `2^d` nodes
// If we perform one add per node,
//   then we will perform O(n) adds on a single traversal of the tree
// The tree we build is not an actual data structure,
//   but a concept we use to determine what each thread does at each step of the traversal
// In this work-efficient scan algorithm,
//   we perform the operations in place on an array in shared memory
// The algorithm consists of two phases: the reduce phase (also known as the up-sweep phase) and the down-sweep phase

#include <iostream>
#include <vector>
#include <numeric>
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

// In the reduce phase (up-sweep phase),
//   we traverse the tree from leaves to root computing partial sums at internal nodes of the tree
// This is also known as a parallel reduction, because after this phase,
//   the root node (the last node in the array) holds the sum of all nodes in the array
__global__ void up_sweep(int* in_arr, int n) {
    int thread_x = threadIdx.x;
    for (int offset = 1; offset < n; offset *= 2) {
        int thread_step = offset * blockDim.x * 2;
        for (int k = (thread_x + 1) * offset * 2 - 1; k < n; k += thread_step) {
            in_arr[k] += in_arr[k - offset];
        }
        __syncthreads();
    }
}

// In the down-sweep phase,
//   we traverse back down the tree from the root,
//   using the partial sums from the reduce phase to build the scan in place on the array
// We start by inserting zero at the root of the tree,
//   and on each step, each node at the current level passes its own value to its left child,
//   and the sum of its value and the former value of its left child to its right child
__global__ void down_sweep(int* block_prefix_sum, int n) {
    int thread_x = threadIdx.x;
    if (thread_x == 0) { block_prefix_sum[n - 1] = 0; }
    for (int offset = n / 2; offset >= 1; offset /= 2) {
        int thread_step = offset * blockDim.x * 2;
        for (int k = (thread_x + 1) * offset * 2 - 1; k < n; k += thread_step) {
            int temp = block_prefix_sum[k];
            block_prefix_sum[k] += block_prefix_sum[k - offset];
            block_prefix_sum[k - offset] = temp;
        }
        __syncthreads();
    }
}


int main() {
    int n = 8;
    int *arr;
    cudaMallocManaged(&arr, n * sizeof(int));

    init_array<<<1, 1>>>(arr);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; ++i) { printf("%d ", arr[i]); }
    printf("\n");

    up_sweep<<<1, 4>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; ++i) { printf("%d ", arr[i]); }

    int sum = arr[n - 1];
    printf("\n");
    down_sweep<<<1, 4>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 1; i < n; ++i) { arr[i - 1] = arr[i]; }
    arr[n - 1] = sum;
    for (int i = 0; i < n; ++i) { printf("%d ", arr[i]); }

    cudaFree(arr);
    return 0;
}
