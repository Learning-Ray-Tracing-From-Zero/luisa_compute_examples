cmake_minimum_required(VERSION 3.26...3.29)


add_executable(cpu cpu.cpp)
target_include_directories(cpu PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include)
target_link_libraries(cpu PRIVATE luisa::compute)

add_executable(cpu2 cpu2.cpp)
target_include_directories(cpu2 PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include)
target_link_libraries(cpu2 PRIVATE luisa::compute)

add_executable(gaussian gaussian.cpp)
target_include_directories(gaussian PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include)
target_link_libraries(gaussian PRIVATE luisa::compute)
