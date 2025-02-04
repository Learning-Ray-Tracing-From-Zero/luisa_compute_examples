cmake_minimum_required(VERSION 3.26...3.29)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if (PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(
        FATAL_ERROR
        "The binary directory of CMake cannot be the same as source directory!"
        "use 'cmake -B build' to specify a different binary directory."
    )
endif()

project(
    HelloLuisaCompute
    VERSION 0.0.1
    DESCRIPTION "Learn LuisaCompute"
    HOMEPAGE_URL "https://github.com/Learning-Ray-Tracing-From-Zero/luisa_compute_examples"
    LANGUAGES C CXX
)


# you may set the LUISA_COMPUTE_ENABLE_XXX options before including LuisaCompute
add_subdirectory(thirdparty/LuisaCompute)
# setup the output directories to make the executables under the same binary directory as LuisaCompute
include(thirdparty/LuisaCompute/scripts/setup_output_dirs.cmake)

add_subdirectory(src)
