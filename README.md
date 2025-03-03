# Luisa Compute Examples 

## Build
build all examples:
```bash
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

bun an example:
```bash
./build/build_release/bin/<programme> <backend>
```

## 1. [CMake Starter Template](./src/cmake_starter_template)
A starter template for using LuisaCompute as a submodule with the CMake build system.

reference: [CMakeStarterTemplate](https://github.com/LuisaGroup/CMakeStarterTemplate)


## 2. [Array Multiplication](./src/array_multiplication)
Multiply the corresponding elements of two arrays and write the result to the third array.


## 3. [Image Post-Processing Algorithm](./src/image_post_processing)

### blur algorithm
+ [box](./src/image_post_processing/blur/box)
+ [gaussian](./src/image_post_processing/blur/gaussian)
+ [kawase](./src/image_post_processing/blur/kawase)


## 4. Path Tracing Algorithm

+ [path tracing with no window](./src/path_tracing/path_tracing.cpp)
