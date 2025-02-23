# LuisaCompute
# you may set the LUISA_COMPUTE_ENABLE_XXX options before including LuisaCompute
set(LuisaCompute_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LuisaCompute CACHE INTERNAL "")
add_subdirectory(LuisaCompute)


# glfw
set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LuisaCompute/src/ext/glfw CACHE INTERNAL "")
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
# add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
add_library(glfw-include INTERFACE)
target_include_directories(glfw-include INTERFACE ${GLFW_DIR}/include)


# DearImgui
set(DEAR_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LuisaCompute/src/ext/imgui CACHE INTERNAL "")
add_library(dear-imgui-include INTERFACE)
target_include_directories(dear-imgui-include INTERFACE ${DEAR_IMGUI_DIR})


# stb
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LuisaCompute/src/ext/stb CACHE INTERNAL "")
