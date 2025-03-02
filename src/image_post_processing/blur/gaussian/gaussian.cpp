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

double gaussian(int x, int y, double sigma = 2.0) {
    double sigma_coefficient = 1.0 / (2.0 * sigma * sigma);
    return std::exp(-(x * x + y * y) * sigma_coefficient);
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <backend> <input_image_path> <output_image_path> <blur_radius> <blur_sigma>\n";
        return 1;
    }

    const char* input_image_path = argv[2];
    const char* output_image_path = argv[3];
    int blur_radius = std::stoi(argv[4]);
    double blur_sigma = std::stod(argv[5]);

    // Step 1: Create a context and device
    Context context { argv[0] };
    Device device = context.create_device(argv[1]);

    // Step 2: Load the input image
    int width { 0 };
    int height { 0 };
    int channels { 0 };
    uint8_t *image_data = stbi_load(input_image_path, &width, &height, &channels, 4); // Force 4 channels (RGBA)
    if (!image_data) {
        std::cerr << "Failed to load image: " << input_image_path << "\n";
        return 1;
    }

    // Step 3: Create images on the device
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    stream << device_image.copy_from(image_data) << synchronize();
    stbi_image_free(image_data);

    Image<float> blurred_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);

    // Step 4: Transfer the weight matrix to the device
    uint kernel_size = blur_radius * 2u + 1u;
    std::vector<float> gaussian_kernel(kernel_size * kernel_size, 0.0f);
    auto kernel_sum { 0.0f };
    for (uint y { 0uz }; y < kernel_size; ++y) {
        for (uint x { 0uz }; x < kernel_size; ++x) {
            gaussian_kernel[x + y * kernel_size] = gaussian(
                x - blur_radius,
                y - blur_radius,
                blur_sigma
            );
            kernel_sum += gaussian_kernel[x + y * kernel_size];
        }
    }
    for (auto& weight : gaussian_kernel) { weight /= kernel_sum; }
    Buffer<float> weight_matrix = device.create_buffer<float>(gaussian_kernel.size());
    stream << weight_matrix.copy_from(gaussian_kernel.data()) << synchronize();

    // Step 5: Define the Gaussian blur kernel
    Kernel2D gaussian_blur_kernel = [&](
        ImageFloat input,
        ImageFloat output,
        BufferFloat weight_matrix
    ) noexcept {
        auto coord = dispatch_id().xy();
        Float4 sum = make_float4(0.0f);
        $for (y, 0u, kernel_size) {
            $for (x, 0u, kernel_size) {
                auto weight = weight_matrix.read(x + y * kernel_size);
                UInt2 sample_coord = make_uint2(0u, 0u);
                Float2 temp_coord = make_float2(
                    Float(coord.x) + Float(x) - Float(blur_radius),
                    Float(coord.y) + Float(y) - Float(blur_radius)
                );
                // processing image boundaries(BORDER_REFLECT_101)
                sample_coord = make_uint2(UInt(abs(temp_coord.x)), UInt(abs(temp_coord.y)));
                $if (temp_coord.x >= Float(dispatch_size().x)) {
                    sample_coord.x = UInt(abs(Float(2u * coord.x) - temp_coord.x));
                };
                $if (temp_coord.y >= Float(dispatch_size().y)) {
                    sample_coord.y = UInt(abs(Float(2u * coord.y) - temp_coord.y));
                };
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
    stbi_write_png(output_image_path, width, height, 4, download_image.data(), 0);

    return 0;
}
