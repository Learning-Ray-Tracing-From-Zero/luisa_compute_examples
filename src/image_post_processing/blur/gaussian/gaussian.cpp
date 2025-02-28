#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <stb/stb_image.h> // stb for image loading
#include <stb/stb_image_write.h> // stb for image saving

#include <iostream>
#include <vector>
#include <array>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;


static constexpr auto PI { 3.14159265358979323846 };

double gaussian(int x, int y, double sigma = 1.5) {
    double sigma_coefficient = 1.0 / 2.0 * sigma * sigma;
    return (sigma_coefficient / PI) * std::exp(-(x * x + y * y) * sigma_coefficient);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <backend> <input_image_path> <blur_kernel_radius>\n";
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

    // Step 3: Create images on the device
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    stream << device_image.copy_from(image_data) << synchronize();

    Image<float> blurred_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);

    // Step 4: Transfer the weight matrix to the device
    constexpr uint KERNEL_RADIUS = argv[3];
    constexpr uint KERNEL_SIZE = KERNEL_RADIUS * 2u + 1u; // Gaussian kernel size (e.g., 5x5)
    constexpr std::array<float, KERNEL_SIZE * KERNEL_SIZE> GAUSSIAN_KERNEL = {};
    for (uint y { 0uz }, y < KERNEL_SIZE; ++y) {
        for (uint x { 0uz }; x < KERNEL_SIZE, ++x) {
            GAUSSIAN_KERNEL[x + y * KERNEL_SIZE] = gaussian(x - KERNEL_RADIUS, y - KERNEL_RADIUS);
        }
    }
    Buffer<float> weight_matrix = device.create_buffer<float>(GAUSSIAN_KERNEL.size());
    stream << weight_matrix.copy_from(GAUSSIAN_KERNEL.data()) << synchronize();

    // Step 5: Define the Gaussian blur kernel
    Kernel2D gaussian_blur_kernel = [](ImageFloat input, ImageFloat output, BufferFloat weight_matrix) noexcept {
        auto coord = dispatch_id().xy();
        auto sum = make_float4(0.0f);
        $for (y, 0u, KERNEL_SIZE) {
            $for (x, 0u, KERNEL_SIZE) {
                auto sample_coord = make_uint2(
                    coord.x - KERNEL_RADIUS + x,
                    coord.y - KERNEL_RADIUS + y
                );
                sample_coord = clamp( // processing image boundaries
                    sample_coord,
                    make_uint2(0u),
                    make_uint2(dispatch_size().xy()) - 1u
                );
                auto weight = weight_matrix_buffer.read(x + y * KERNEL_SIZE);
                sum += input->read(sample_coord) * weight;
            };
        };
        output->write(coord, sum);
    };
    Shader gaussian_blur = device.compile(gaussian_blur_kernel); // compile the kernel

    // Apply the Gaussian blur
    stream << gaussian_blur(device_image.view(0), blurred_image.view(0), weight_matrix).dispatch(width, height)
           << synchronize();

    // Download the blurred image back to the host
    std::vector<std::byte> download_image(width * height * 4);
    stream << blurred_image.copy_to(download_image.data()) << synchronize();

    // Save the blurred image
    stbi_write_png("blurred_output.png", width, height, 4, download_image.data(), 0);
    stbi_image_free(image_data); // clean up

    return 0;
}
