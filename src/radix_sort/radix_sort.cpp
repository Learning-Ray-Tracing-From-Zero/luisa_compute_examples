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
    std::array<value_type, N> in_array;
    std::array<value_type, N> out_array;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<value_type> distrib(0, 0x0FFFFFFF);
    for (std::size_t  i { 0uz }; i < N; ++i) {
        in_array[i] = distrib(gen);
        out_array[i] = static_cast<value_type>(0);
    }

    // CPU std::sort Test
    Clock clock;
    std::sort(in_array.begin(), in_array.end(), std::less<>{});
    LUISA_INFO("CPU uses std::sort sorting time: {}", clock.toc());

    // GPU radix sort Test
    log_level_verbose();
    Context context { argv[0] };
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    Buffer<value_type> in_array_buffer = device.create_buffer<value_type>(N);
    Buffer<value_type> out_array_buffer = device.create_buffer<value_type>(N)
    stream << in_array_buffer.copy_from(in_array.data()) << synchronize();

    Kernel1D kernel = [&](buffer_type in, buffer_type out) {
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
