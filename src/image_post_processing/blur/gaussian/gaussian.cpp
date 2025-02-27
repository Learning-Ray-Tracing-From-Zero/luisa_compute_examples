#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <stb/stb_image.h> // stb for image loading
#include <stb/stb_image_write.h> // stb for image saving

#include <iostream>
#include <vector>

using namespace luisa;
using namespace luisa::compute;


// Gaussian kernel size (e.g., 5x5)
constexpr int KERNEL_RADIUS = 2;
constexpr int KERNEL_SIZE = KERNEL_RADIUS * 2 + 1;

// Precomputed Gaussian kernel weights (normalized)
constexpr float GAUSSIAN_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
    { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f },
    { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
    { 0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f },
    { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
    { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f }
};


int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <backend> <input_image_path>\n";
        return 1;
    }

    // Step 1: Create a context and device
    Context context { argv[0] };
    Device device = context.create_device(argv[1]);

    // Step 2: Load the input image
    int width { 0 };
    int height { 0 };
    int channels { 0 };
    uint8_t *image_data = stbi_load(argv[2], &width, &height, &channels, 4); // Force 4 channels (RGBA)
    if (!image_data) {
        std::cerr << "Failed to load image: " << argv[2] << "\n";
        return 1;
    }

    // Step 3: Create an image on the device
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    stream << device_image.copy_from(image_data) << synchronize();

    // Step 4: Define the Gaussian blur kernel
    Kernel2D gaussian_blur_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        Var coord = dispatch_id().xy();
        Var sum = make_float4(0.0f);
        $for (int i : range(-KERNEL_RADIUS, KERNEL_RADIUS + 1)) {
            $for (int j : range(-KERNEL_RADIUS, KERNEL_RADIUS + 1)) {
                Var sample_coord = coord + make_int2(i, j);
                sample_coord = clamp(sample_coord, make_int2(0), make_int2(dispatch_size().xy()) - 1);
                Var weight = GAUSSIAN_KERNEL[i + KERNEL_RADIUS][j + KERNEL_RADIUS];
                sum += input->read(sample_coord) * weight;
            }
        };
        output->write(coord, sum);
    };
    Shader gaussian_blur = device.compile(gaussian_blur_kernel); // compile the kernel

    // Create an output image on the device
    Image<float> blurred_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);

    // Apply the Gaussian blur
    stream << gaussian_blur(device_image.view(0), blurred_image.view(0)).dispatch(width, height)
           << synchronize();

    // Download the blurred image back to the host
    std::vector<std::byte> download_image(width * height * 4);
    stream << blurred_image.copy_to(download_image.data())
           << synchronize();

    // Save the blurred image
    stbi_write_png("blurred_output.png", width, height, 4, download_image.data(), 0);
    stbi_image_free(image_data); // clean up

    return 0;
}
