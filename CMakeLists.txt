cmake_minimum_required(VERSION 3.26...3.29)

cmake_policy(VERSION 3.26)
set(CMAKE_POLICY_DEFAULT_CMP0042 NEW) # enable MACOSX_RPATH, which means using @rpath by default on macOS
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW) # new behavior of enabling variables such as PROJECT_VERSION
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW) # prohibit the use of undefined symbols in add_ibrary()
set(CMAKE_POLICY_DEFAULT_CMP0068 NEW) # enable RPATH support
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW) # enable INTERPROCEDURAL_OPTIMIZATION(IPO) support
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW) # find_package() prioritizes using Config mode
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW) # allow the use of cache variables in option()
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW) # enable new MSVC runtime library selection logic
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW) # new behavior to enable file(GENERATE)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
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

# enable the use of folder structure in the project
# it will organize files according to the folder structure in the generated IDE project (such as Visual Studio or Xcode)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(thirdparty)

add_subdirectory(src)
