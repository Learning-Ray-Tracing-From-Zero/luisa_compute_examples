#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

// Each iteration uses different buffers for reading and writing,
//   thereby avoiding concurrent reads and writes to the same memory location in the same iteration,
//   simplifying data dependency management, and potentially improving performance

// CUDA kernel for exclusive prefix sum within a block
// in_arr: input array
// out_arr: output array
// n: the size of data processed by each thread block
__global__ void kernel(float* in_arr, float* out_arr, int n) {
    // extern: indicates dynamically allocating its size during kernel calls
    // __shared__: shared within a single thread block
    extern __shared__ float shared_arr[];
    int thread_x = threadIdx.x;
    int thread_id = blockIdx.x * blockDim.x + thread_x;
    int previous_out = 0;
    int previous_in = 1;

    // Load input into shared memory (exclusive scan: shift right by one, first element is 0)
    shared_arr[previous_out * n + thread_x] = (thread_x > 0 && thread_x < n) ? in_arr[thread_id - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        previous_out = 1 - previous_out; // swap double buffer indices
        previous_in = 1 - previous_out;
        int index_pout = previous_out * n + thread_x;
        int index_pin = previous_in * n + thread_x;
        if (thread_x >= offset && thread_x < n) {
            shared_arr[index_pout] = shared_arr[index_pin] + shared_arr[index_pin - offset];
        } else if (thread_x < n) {
            shared_arr[index_pout] = shared_arr[index_pin];
        }
        __syncthreads();
    }

    // Write output
    if (thread_x < n) { out_arr[thread_id] = shared_arr[previous_out * n + thread_x]; }
}


int main() {
    int n = 16; // size of the array (must be equal to blockDim.x for this simplified example)
    std::vector<float> in_arr(n);
    std::vector<float> exclusive_prefix_sum(n);
    std::vector<float> inclusive_prefix_sum(n);

    // Initialize input data on the host
    for (int i = 0; i < n; ++i) { in_arr[i] = static_cast<float>(i + 1); }

    // Allocate device memory
    float* in_arr_data = nullptr;
    float* out_arr_data = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&in_arr_data, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for input: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&out_arr_data, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for output: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(in_arr_data);
        return 1;
    }

    // Copy input data to device
    cudaStatus = cudaMemcpy(in_arr_data, in_arr.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed to copy input to device: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(in_arr_data);
        cudaFree(out_arr_data);
        return 1;
    }

    // Launch the CUDA kernel
    int grid_size = 1;
    int block_size = n;
    int shared_memory_size = 2 * block_size * sizeof(float);
    kernel<<<grid_size, block_size, shared_memory_size>>>(in_arr_data, out_arr_data, block_size);

    // Wait for kernel execution to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus
                  << " after launching kernel: " << cudaGetErrorString(cudaStatus)
                  << std::endl;
        cudaFree(in_arr_data);
        cudaFree(out_arr_data);
        return 1;
    }

    // Copy output data from device to host
    cudaStatus = cudaMemcpy(exclusive_prefix_sum.data(), out_arr_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed to copy output from device: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(in_arr_data);
        cudaFree(out_arr_data);
        return 1;
    }

    std::cout << "exclusive prefix sum: ";
    for (float val : exclusive_prefix_sum) { std::cout << val << " "; }
    std::cout << std::endl;

    std::cout << "inclusive prefix sum: ";
    std::copy(
        exclusive_prefix_sum.begin() + 1,
        exclusive_prefix_sum.end(),
        inclusive_prefix_sum.begin()
    );
    inclusive_prefix_sum[n - 1] = inclusive_prefix_sum[n - 2] + in_arr[n - 1];
    for (float val : inclusive_prefix_sum) { std::cout << val << " "; }

    // Free device memory
    cudaFree(in_arr_data);
    cudaFree(out_arr_data);

    return 0;
}
