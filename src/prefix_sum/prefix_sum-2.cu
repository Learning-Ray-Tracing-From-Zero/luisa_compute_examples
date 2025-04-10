#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>


// CUDA kernel for exclusive prefix sum within a block
__global__ void kernel(float* in_arr, float* out_arr, int n) {
    extern __shared__ float shared_arr[]; // allocated on invocation
    int thread_x = threadIdx.x;
    int thread_id = blockIdx.x * blockDim.x + thread_x;
    int pout = 0;
    int pin = 1;

    // Load input into shared memory (exclusive scan: shift right by one, first element is 0)
    shared_arr[pout * n + thread_x] = (thread_x > 0 && thread_x < n) ? in_arr[thread_id - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        int index_pout = pout * n + thread_x;
        int index_pin = pin * n + thread_x;
        if (thread_x >= offset && thread_x < n) {
            shared_arr[index_pout] = shared_arr[index_pin] + shared_arr[index_pin - offset];
        } else if (thread_x < n) {
            shared_arr[index_pout] = shared_arr[index_pin];
        }
        __syncthreads();
    }

    // Write output
    if (thread_x < n) { out_arr[thread_id] = shared_arr[pout * n + thread_x]; }
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

    // Configure thread block and grid size
    int blockSize = n;
    int gridSize = 1;

    // Launch the CUDA kernel
    kernel<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(in_arr_data, out_arr_data, blockSize);

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
