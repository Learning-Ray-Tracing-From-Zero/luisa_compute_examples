cmake_minimum_required(VERSION 3.26...3.29)


add_executable(cmake_starter_template main.cpp)
target_include_directories(
    cmake_starter_template
    PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(
    cmake_starter_template
    PRIVATE
    luisa::compute
)
