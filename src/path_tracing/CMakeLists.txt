add_executable(path_tracing path_tracing.cpp)
target_include_directories(
    path_tracing
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(path_tracing PRIVATE luisa::compute)
