#include <stb/stb_image.h> // stb for image loading
#include <stb/stb_image_write.h> // stb for image saving

#include <iostream>
#include <vector>
#include <array>
#include <cmath>


static constexpr auto PI { 3.14159265358979323846 };

double gaussian(int x, int y, double sigma = 2.0) {
    double sigma_coefficient = 1.0 / (2.0 * sigma * sigma);
    return (sigma_coefficient / PI) * std::exp(-(x * x + y * y) * sigma_coefficient);
}


struct Pixel {
    float r { 0.0f };
    float g { 0.0f };
    float b { 0.0f };
};


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: "
                  << argv[0] << " <input_image_path> <output_image_path> <blur_radius> <blur_sigma>\n";
        return 1;
    }

    const char* input_image_path = argv[1];
    const char* output_image_path = argv[2];
    int blur_radius = std::stoi(argv[3]);
    double blur_sigma = std::stod(argv[4]);

    int width { 0 };
    int height { 0 };
    int channels { 0 };
    uint8_t *image_data = stbi_load(input_image_path, &width, &height, &channels, 3);
    if (!image_data) {
        std::cerr << "Failed to load image: " << input_image_path << "\n";
        return 1;
    }
    
    std::vector<Pixel> input_image(width * height);
    for (std::size_t i { 0uz }; i < input_image.size(); ++i) {
        input_image[i] = {
            .r = static_cast<float>(image_data[channels * i + 0]),
            .g = static_cast<float>(image_data[channels * i + 1]),
            .b = static_cast<float>(image_data[channels * i + 2])
        };
    }
    stbi_image_free(image_data);

    std::vector<std::uint8_t> output_image(width * height * channels, 0);

    auto w { 0.0f };
    int kernel_size = blur_radius * 2u + 1u;
    std::vector<float> GAUSSIAN_KERNEL(kernel_size * kernel_size, 0.0f);
    for (int y { 0 }; y < kernel_size; ++y) {
        for (int x { 0 }; x < kernel_size; ++x) {
            GAUSSIAN_KERNEL[x + y * kernel_size] = gaussian(
                x - blur_radius,
                blur_radius - y,
                blur_sigma
            );
            w += GAUSSIAN_KERNEL[x + y * kernel_size];
        }
    }
    for (auto &i : GAUSSIAN_KERNEL) { i /= w; }

    for (int y { 0 }; y < height; ++y) {
        for (int x { 0 }; x < width; ++x) {
            auto sum_r { 0.0f };
            auto sum_g { 0.0f };
            auto sum_b { 0.0f };
            float temp { 0.0f };
            for (int j { 0uz }; j < kernel_size; ++j) {
                for (int i { 0uz }; i < kernel_size; ++i) {
                    int sample_coord_x = std::clamp(x - blur_radius + i, 0, width - 1);
                    int sample_coord_y = std::clamp(y - blur_radius + j, 0, height - 1);
                    int pixel_index = sample_coord_x + sample_coord_y * width;
                    auto weight = GAUSSIAN_KERNEL[i + j * kernel_size];
                    sum_r += input_image[pixel_index].r * weight;
                    sum_g += input_image[pixel_index].g * weight;
                    sum_b += input_image[pixel_index].b * weight;
                    temp += weight;
                }
            }
            int output_pixel_index = channels * (x + y * width);
            output_image[output_pixel_index + 0] = static_cast<std::uint8_t>(sum_r);
            output_image[output_pixel_index + 1] = static_cast<std::uint8_t>(sum_g);
            output_image[output_pixel_index + 2] = static_cast<std::uint8_t>(sum_b);
        }
    }

    if (!stbi_write_png(output_image_path, width, height, channels, output_image.data(), 0)) {
        std::cerr << "Failed to save image: " << output_image_path << "\n";
        return 1;
    }
    std::cout << "Gaussian blur applied successfully.\nOutput saved to: " << output_image_path << "\n";

    return 0;
}
