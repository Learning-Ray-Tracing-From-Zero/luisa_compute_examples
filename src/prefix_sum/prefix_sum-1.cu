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

__global__ void kernel(int* arr, int n, int d) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= static_cast<int>(pow(2, d - 1)) && k < n) {
        arr[k] = arr[k - static_cast<int>(pow(2, d - 1))] + arr[k];
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

    for (int d = 1; d <= static_cast<int>(log2(n)); ++d) {
        kernel<<<8, 1, 1>>>(arr, n, d);
        cudaDeviceSynchronize();
        for (int i = 0; i < n; i++) { printf("%d ", arr[i]); }
        printf("\n");
    }

    cudaFree(arr);
    return 0;
}
