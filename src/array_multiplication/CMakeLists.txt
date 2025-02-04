cmake_minimum_required(VERSION 3.26...3.29)


add_executable(array_multiplication array_multiplication.cpp)
target_include_directories(
    array_multiplication
    PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(
    array_multiplication
    PRIVATE
    luisa::compute
)
