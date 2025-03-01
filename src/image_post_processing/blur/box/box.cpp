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


double box_distribution(int kernel_size) {
    return 1.0 / (kernel_size * kernel_size);
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <backend> <input_image_path> <output_image_path> <blur_radius>\n";
        return 1;
    }

    const char* input_image_path = argv[2];
    const char* output_image_path = argv[3];
    int blur_radius = std::stoi(argv[4]);

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
    std::vector<float> box_weight(kernel_size * kernel_size, 0.0f);
    for (uint y { 0uz }; y < kernel_size; ++y) {
        for (uint x { 0uz }; x < kernel_size; ++x) {
            box_weight[x + y * kernel_size] = box_distribution(kernel_size);
        }
    }
    Buffer<float> weight_matrix = device.create_buffer<float>(box_weight.size());
    stream << weight_matrix.copy_from(box_weight.data()) << synchronize();

    // Step 5: Define the Box blur kernel
    Kernel2D box_blur_kernel = [&](
        ImageFloat input,
        ImageFloat output,
        BufferFloat weight_matrix
    ) noexcept {
        auto coord = dispatch_id().xy();
        Float4 sum = make_float4(0.0f);
        $for (y, 0u, kernel_size) {
            $for (x, 0u, kernel_size) {
                auto sample_coord = make_uint2(
                    coord.x - blur_radius + x,
                    coord.y - blur_radius + y
                );
                sample_coord = clamp( // processing image boundaries
                    sample_coord,
                    make_uint2(0u),
                    make_uint2(dispatch_size().xy()) - 1u
                );
                auto weight = weight_matrix.read(x + y * kernel_size);
                sum += input->read(sample_coord) * weight;
            };
        };
        output->write(coord, sum);
    };
    Shader box_blur = device.compile(box_blur_kernel); // compile the kernel

    // Apply the Box blur
    stream << box_blur(device_image.view(0), blurred_image.view(0), weight_matrix).dispatch(width, height)
           << synchronize();

    // Download the blurred image back to the host
    std::vector<std::byte> download_image(width * height * 4);
    stream << blurred_image.copy_to(download_image.data()) << synchronize();

    // Save the blurred image
    stbi_write_png(output_image_path, width, height, 4, download_image.data(), 0);

    return 0;
}
