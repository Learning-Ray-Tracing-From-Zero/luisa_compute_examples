#include <iostream>
#include <cuda_runtime.h>


int main() {
    int device_id = 0;
    cudaDeviceProp props;
    if (cudaError_t error = cudaGetDeviceProperties(&props, device_id); error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "Maximum shared memory per block on device " << device_id << ": "
              << props.sharedMemPerBlock / 1024.0 << " KB" << std::endl;

    return 0;
}
