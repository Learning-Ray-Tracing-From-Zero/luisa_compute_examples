add_executable(ray_tracing_in_one_weekend main.cpp)
target_include_directories(
    ray_tracing_in_one_weekend
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/ray_tracing_in_one_weekend/include
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cxxopts/include
)
target_link_libraries(ray_tracing_in_one_weekend PRIVATE luisa::compute)
