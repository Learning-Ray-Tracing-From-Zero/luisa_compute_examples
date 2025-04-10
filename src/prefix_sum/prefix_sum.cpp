#include <array>
#include <cstdint>
#include <random>
#include <algorithm> // std::copy
#include <iostream>


int main() {
    using size_type = std::size_t;
    using value_type = std::uint32_t;

    constexpr size_type N { 8uz };
    std::array<value_type, N> arr;
    std::array<value_type, N> exclusive_prefix_sum;
    std::array<value_type, N> inclusive_prefix_sum;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<value_type> distrib(0, 0x0FFFFFFF);
    for (std::size_t  i { 0uz }; i < N; ++i) { arr[i] = distrib(gen); }
    exclusive_prefix_sum.fill(static_cast<value_type>(0));
    inclusive_prefix_sum.fill(static_cast<value_type>(0));

    // CPU Test
    for (size_type i { 1uz }; i < arr.size(); ++i) {
        std::size_t k = i - 1uz;
        exclusive_prefix_sum[i] = exclusive_prefix_sum[k] + arr[k];
    }
    std::copy(
        exclusive_prefix_sum.begin() + 1,
        exclusive_prefix_sum.end(),
        inclusive_prefix_sum.begin()
    );
    inclusive_prefix_sum[N - 1uz] = inclusive_prefix_sum[N - 2uz] + arr[N - 1uz];

    for (const auto& i : arr) { std::cout << i << " "; }
    std::cout << std::endl;
    for (const auto& i : exclusive_prefix_sum) { std::cout << i << " "; }
    std::cout << std::endl;
    for (const auto& i : inclusive_prefix_sum) { std::cout << i << " "; }

    return 0;
}
