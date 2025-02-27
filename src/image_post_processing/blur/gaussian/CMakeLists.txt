cmake_minimum_required(VERSION 3.26...3.29)


add_executable(gaussian gaussian.cpp)
target_include_directories(
    gaussian
    PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(
    gaussian
    PRIVATE
    luisa::compute
)
