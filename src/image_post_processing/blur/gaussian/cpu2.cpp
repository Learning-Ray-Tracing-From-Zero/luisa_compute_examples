#include <stb/stb_image.h> // stb for image loading
#include <stb/stb_image_write.h> // stb for image saving

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // std::clamp


static constexpr auto PI { 3.14159265358979323846 };

double gaussian(int x, int y, double sigma = 2.0) {
    double sigma_coefficient = 1.0 / (2.0 * sigma * sigma);
    // return (sigma_coefficient / PI) * std::exp(-(x * x + y * y) * sigma_coefficient);
    return std::exp(-(x * x + y * y) * sigma_coefficient);
}


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

    if (blur_radius <= 0 || blur_sigma <= 0) {
        std::cerr << "Error: Blur radius and sigma must be positive values.\n";
        return 1;
    }

    int width { 0 };
    int height { 0 };
    int channels { 0 };
    std::uint8_t* image_data = stbi_load(input_image_path, &width, &height, &channels, 3);
    if (!image_data) {
        std::cerr << "Failed to load image: " << input_image_path << "\n";
        return 1;
    }

    std::vector<std::uint8_t> input_image(image_data, image_data + width * height * channels);
    std::vector<std::uint8_t> output_image(width * height * channels, 0);
    stbi_image_free(image_data);

    int kernel_size = blur_radius * 2 + 1;
    std::vector<float> gaussian_kernel(kernel_size * kernel_size, 0.0f);
    auto kernel_sum { 0.0f };
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            gaussian_kernel[x + y * kernel_size] = gaussian(
                x - blur_radius,
                blur_radius - y, // 'y - blur_radius' is also ok
                blur_sigma
            );
            kernel_sum += gaussian_kernel[x + y * kernel_size];
        }
    }
    for (auto& weight : gaussian_kernel) { weight /= kernel_sum; }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto sum_r { 0.0f };
            auto sum_g { 0.0f };
            auto sum_b { 0.0f };
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int sample_x = std::clamp(x - blur_radius + kx, 0, width - 1);
                    int sample_y = std::clamp(y - blur_radius + ky, 0, height - 1);
                    int pixel_index = channels * (sample_x + sample_y * width);
                    float weight = gaussian_kernel[kx + ky * kernel_size];
                    sum_r += input_image[pixel_index + 0] * weight;
                    sum_g += input_image[pixel_index + 1] * weight;
                    sum_b += input_image[pixel_index + 2] * weight;
                }
            }
            int output_pixel_index = channels * (x + y * width);
            output_image[output_pixel_index + 0] = static_cast<uint8_t>(sum_r);
            output_image[output_pixel_index + 1] = static_cast<uint8_t>(sum_g);
            output_image[output_pixel_index + 2] = static_cast<uint8_t>(sum_b);
        }
    }

    if (!stbi_write_png(output_image_path, width, height, channels, output_image.data(), 0)) {
        std::cerr << "Failed to save image: " << output_image_path << "\n";
        return 1;
    }
    std::cout << "Gaussian blur applied successfully.\nOutput saved to: " << output_image_path << "\n";

    return 0;
}
