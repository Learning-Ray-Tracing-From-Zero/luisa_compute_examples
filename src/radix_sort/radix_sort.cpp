#include <luisa/luisa-compute.h>

#include <cstdint> // std::uint32_t
#include <array> // std::array
#include <random> // std::random_device, std::mt19937, std::uniform_int_distribution
#include <algorithm> // std::sort
#include <functional> // std::less


int main() {
    using size_type = std::size_t;
    using value_type = std::uint32_t;
    using buffer_type = BufferVar<value_type>;

    constexpr size_type N { 1024uz * 1024uz * 32uz };
    constexpr size_type length = ((N / (256uz * 16uz)) < 1uz) ? 1uz : (N / (256uz * 16uz));
    std::array<size_type, N> key_array;
    std::array<value_type, N> value_array;
    std::array<size_type, N> key_sorted_array;
    std::array<value_type, N> value_sorted_array;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<value_type> distrib(0, 0x0FFFFFFF);
    for (std::size_t  i { 0uz }; i < N; ++i) {
        key_array[i] = N - i;
        value_array[i] = i + 1;
        // value_array[i] = distrib(gen);
    }
    key_sorted_array.fill(static_cast<value_type>(0));
    value_sorted_array.fill(static_cast<value_type>(0));

    // CPU std::sort Test
    Clock clock;
    value_sorted_array = std::move(value_array);
    std::sort(value_sorted_array.begin(), value_sorted_array.end(), std::less<>{});
    LUISA_INFO("CPU uses std::sort sorting time: {}", clock.toc());

    // GPU radix sort Test
    log_level_verbose();
    Context context { argv[0] };
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    std::array<value_type, length> sub_group_sum_array1;
    std::array<value_type, N / 2048uz> sub_group_sum_array2;
    sub_group_sum_array1.fill(static_cast<value_type>(0));
    sub_group_sum_array2.fill(static_cast<value_type>(0));
    Buffer<value_type> key_array_buffer = device.create_buffer<value_type>(N);
    Buffer<value_type> value_array_buffer = device.create_buffer<value_type>(N);
    Buffer<value_type> key_sorted_array_buffer = device.create_buffer<value_type>(N);
    Buffer<value_type> value_sorted_array_buffer = device.create_buffer<value_type>(N);
    stream << key_array_buffer.copy_from(key_array.data())
           << value_array_buffer.copy_from(value_array.data())
           << synchronize();

    Kernel1D array_sum = [&](buffer_type in_arr, buffer_type out_arr) {

    }

    Kernel1D up_sweep = [&]() {
        // TODO
    }

    Kernel1D down_sweep = [&]() {
        // TODO
    }

    Kernel1D prefix_scan = [&]() {
        // TODO
    }

    Kernel1D radix_sort = [&](buffer_type in, buffer_type out) {
        // TODO
        UInt i = dispatch_x();
        out.write(i, in.read(i));
    };
    Shader shader = device.compile(kernel);
    stream << shader(in_array_buffer).dispatch(N)
           << out_array_buffer.copy_to(out_array.data())
           << synchronize();

    return 0;
}
