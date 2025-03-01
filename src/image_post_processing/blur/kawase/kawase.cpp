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


double kawase_distribution(int x, int y, int blur_radius, double base) {
    return (std::pow(2, 2 * blur_radius) / std::pow(2, std::abs(x) + std::abs(y))) / (base * base);
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <backend> <input_image_path> <output_image_path> <iter_size>\n";
        return 1;
    }

    const char* input_image_path = argv[2];
    const char* output_image_path = argv[3];
    int iter_size = std::stoi(argv[4]);

    Context context { argv[0] };
    Device device = context.create_device(argv[1]);

    int width { 0 };
    int height { 0 };
    int channels { 0 };
    uint8_t *image_data = stbi_load(input_image_path, &width, &height, &channels, 4); // Force 4 channels (RGBA)
    if (!image_data) {
        std::cerr << "Failed to load image: " << input_image_path << "\n";
        return 1;
    }

    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    stream << device_image.copy_from(image_data) << synchronize();
    stbi_image_free(image_data);

    Image<float> blurred_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    for (int blur_radius { 1uz }; blur_radius < iter_size + 1; ++blur_radius) {
        uint kernel_size = blur_radius * 2u + 1u;
        std::vector<float> kawase_weight(kernel_size * kernel_size, 0.0f);
        for (uint y { 0uz }; y < kernel_size; ++y) {
            for (uint x { 0uz }; x < kernel_size; ++x) {
                kawase_weight[x + y * kernel_size] = kawase_distribution(
                    x - blur_radius,
                    y - blur_radius,
                    blur_radius,
                    3.0 * std::pow(2, blur_radius) - 2.0
                );
            }
        }
        Buffer<float> weight_matrix = device.create_buffer<float>(kawase_weight.size());
        stream << weight_matrix.copy_from(kawase_weight.data()) << synchronize();

        Kernel2D kawase_blur_kernel = [&](
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
                    sample_coord = clamp(
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
        Shader kawase_blur = device.compile(kawase_blur_kernel);

        Image<float> temp_image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
        stream << device_image.copy_to(temp_image) << synchronize();
        stream << kawase_blur(temp_image.view(0), blurred_image.view(0), weight_matrix).dispatch(width, height)
               << synchronize();
        stream << blurred_image.copy_to(device_image) << synchronize();
    }

    std::vector<std::byte> download_image(width * height * 4);
    stream << blurred_image.copy_to(download_image.data()) << synchronize();

    stbi_write_png(output_image_path, width, height, 4, download_image.data(), 0);

    return 0;
}
