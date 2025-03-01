add_executable(box box.cpp)
target_include_directories(box PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include)
target_link_libraries(box PRIVATE luisa::compute)
