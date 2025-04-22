# vulkan
find_package(Vulkan REQUIRED)
include_directories(${GLFW_DIR}/deps)

message("GLFW_DIR: ${GLFW_DIR}")
message("DEAR_IMGUI_DIR: ${DEAR_IMGUI_DIR}")
add_executable(
    dear_imgui
    test_dear_imgui.cpp
    ${DEAR_IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${DEAR_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${DEAR_IMGUI_DIR}/imgui.cpp
    ${DEAR_IMGUI_DIR}/imgui_draw.cpp
    ${DEAR_IMGUI_DIR}/imgui_demo.cpp
    ${DEAR_IMGUI_DIR}/imgui_tables.cpp
    ${DEAR_IMGUI_DIR}/imgui_widgets.cpp
)
target_include_directories(
    dear_imgui
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(
    dear_imgui
    PRIVATE
    glfw
    glfw-include
    dear-imgui-include
    Vulkan::Vulkan
    luisa::compute
)
