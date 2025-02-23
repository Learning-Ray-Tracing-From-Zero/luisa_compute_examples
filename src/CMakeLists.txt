# setup the output directories to make the executables under the same binary directory as LuisaCompute
include(${LuisaCompute_DIR}/scripts/setup_output_dirs.cmake)

add_subdirectory(cmake_starter_template)
add_subdirectory(array_multiplication)
# add_subdirectory(dear_imgui)