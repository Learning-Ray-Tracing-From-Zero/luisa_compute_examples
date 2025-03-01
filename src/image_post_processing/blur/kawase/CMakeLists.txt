add_executable(kawase kawase.cpp)
target_include_directories(kawase PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include)
target_link_libraries(kawase PRIVATE luisa::compute)
