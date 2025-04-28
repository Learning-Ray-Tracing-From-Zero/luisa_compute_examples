add_executable(
    path_tracing_gui
    ./path_tracing_gui.cpp

    ./camera/camera_control.hpp
    ./camera/camera.hpp

    ./core/application.hpp
    ./core/application.cpp
    ./core/window.hpp

    ./gui/application_event.hpp
    ./gui/event.hpp
    ./gui/key_codes.hpp
    ./gui/key_event.hpp
    ./gui/mouse_codes.hpp
    ./gui/mouse_event.hpp
    ./gui/windows_window.hpp
    ./gui/windows_window.cpp
)
target_include_directories(
    path_tracing_gui
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/path_tracing_gui
    ${CMAKE_SOURCE_DIR}/thirdparty/LuisaCompute/include
)
target_link_libraries(path_tracing_gui PRIVATE luisa::compute)
