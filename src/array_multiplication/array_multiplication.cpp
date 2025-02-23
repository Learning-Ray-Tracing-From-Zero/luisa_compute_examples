#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <iostream>
#include <string>
#include <array>

using namespace luisa;
using namespace luisa::compute;


void print_array(
    const std::string& name,
    const std::array<float, 3uz>& arr
) {
    std::cout << "array " << name << ": ";
    for (const auto i : arr) {
        std::cout << i << " ";
    }
    std::cout << '\n';
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <backend>\n";
        return 1;
    }

    Context context { argv[0] };
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    constexpr std::size_t n { 3uz };
    std::array<float, n> a = { 1.0f, 2.0f, 3.0f };
    std::array<float, n> b = { 4.0f, 5.0f, 6.0f };
    std::array<float, n> c = { 0.0f, 0.0f, 0.0f };
    Buffer<float> buf_a = device.create_buffer<float>(n);
    Buffer<float> buf_b = device.create_buffer<float>(n);
    Buffer<float> buf_c = device.create_buffer<float>(n);
    stream << buf_a.copy_from(a.data())
           << buf_b.copy_from(b.data())
           << synchronize();

    Kernel1D kernel = [&](BufferFloat a, BufferFloat b, BufferFloat c) {
        UInt i = dispatch_x();
        c.write(i, a.read(i) * b.read(i));
    };
    Shader shader = device.compile(kernel);
    stream << shader(buf_a, buf_b, buf_c).dispatch(n)
           << buf_c.copy_to(c.data())
           << synchronize();

    print_array("a", a);
    print_array("b", b);
    print_array("c", c);

    return 0;
}
