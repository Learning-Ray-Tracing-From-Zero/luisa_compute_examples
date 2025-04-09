add_executable(radix_sort radix_sort.cpp)
target_include_directories(
    radix_sort
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(radix_sort PRIVATE luisa::compute)
