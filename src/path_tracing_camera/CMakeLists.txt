add_executable(
    path_tracing_camera
    path_tracing_camera.cpp
)
target_include_directories(
    path_tracing_camera
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(path_tracing_camera PRIVATE luisa::compute)
