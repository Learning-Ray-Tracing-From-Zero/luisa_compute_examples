#include <gui/windows_window.hpp>
#include <gui/application_event.hpp>
#include <gui/key_event.hpp>
#include <gui/mouse_event.hpp>

#include <iostream>


static auto s_glfw_initialized { false };
static void glfw_error_callback(int error, const char* description) {
    std::cout << "GLFW Error: (" << error << " : " << description << ")" << std::endl;
}

Window* Window::create(const WindowProps &props) {
    return new WindowsWindow(props);
}

WindowsWindow::WindowsWindow(const WindowProps &props) {
    init(props);
}

WindowsWindow::~WindowsWindow() noexcept {
    shutdown();
}

void WindowsWindow::init(const WindowProps &props) {
    _window_data.title = props.title;
    _window_data.width = props.width;
    _window_data.height = props.height;

    std::cout << "Creating window: " << _window_data.title
              << " (" << _window_data.width << ", "<< _window_data.height << ")"
              << std::endl;

    if (!s_glfw_initialized) {
        if (int success = glfwInit(); !success) { return ; }
        glfwSetErrorCallback(glfw_error_callback);

        s_glfw_initialized = true;
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _glfw_window = glfwCreateWindow(
        static_cast<int>(_window_data.width),
        static_cast<int>(_window_data.height),
        _window_data.title.c_str(),
        nullptr,
        nullptr
    );
    glfwMakeContextCurrent(_glfw_window);
    glfwSetWindowUserPointer(_glfw_window, &_window_data);

    // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    // {
    //     HZ_CORE_INFO("Failed to initialize GLAD");
    //     return;
    // }
    set_vsync(true);


    // Set GLFW callback
    glfwSetWindowSizeCallback(
        _glfw_window,
        [] (GLFWwindow* window, int width, int height) {
            WindowResizeEvent window_resize_event(width, height);
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            data->width = width;
            data->height = height;
            data->event_callback(window_resize_event);
        }
    );

    glfwSetWindowCloseCallback(
        _glfw_window,
        [] (GLFWwindow* window){
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            WindowCloseEvent window_close_event;
            data->event_callback(window_close_event);
        }
    );

    glfwSetKeyCallback(
        _glfw_window,
        [] (GLFWwindow* window, int key, int scancode, int action, int mods){
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            switch (action) {
                case GLFW_PRESS: {
                    KeyPressedEvent key_pressed_event(key, false);
                    data->event_callback(key_pressed_event);
                    break;
                }
                case GLFW_RELEASE: {
                    KeyPressedEvent key_pressed_event(key, false);
                    data->event_callback(key_pressed_event);
                    break;
                }
                case GLFW_REPEAT: {
                    KeyPressedEvent key_pressed_event(key, true);
                    data->event_callback(key_pressed_event);
                    break;
                }
            }
        }
    );

    glfwSetMouseButtonCallback(
        _glfw_window,
        [] (GLFWwindow* window, int button, int action, int mods){
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            switch (action) {
                case GLFW_PRESS: {
                    MouseButtonPressedEvent event(button);
                    data->event_callback(event);
                    break;
                }
                case GLFW_RELEASE: {
                    MouseButtonReleasedEvent event(button);
                    data->event_callback(event);
                    break;
                }
            }
        }
    );

    glfwSetScrollCallback(
        _glfw_window,
        [] (GLFWwindow* window, double xOffset, double yOffset){
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            MouseScrolledEvent event(
                static_cast<float>(xOffset),
                static_cast<float>(yOffset)
            );
            data->event_callback(event);
        }
    );

    glfwSetCursorPosCallback(
        _glfw_window,
        [] (GLFWwindow* window, double xOffset, double yOffset){
            WindowData* data = static_cast<WindowData*>(glfwGetWindowUserPointer(window));
            MouseMovedEvent event(
                static_cast<float>(xOffset),
                static_cast<float>(yOffset)
            );
            data->event_callback(event);
        }
    );
}

void WindowsWindow::shutdown() {
    glfwDestroyWindow(_glfw_window);
}

void WindowsWindow::update() {
    glfwPollEvents();
    glfwSwapBuffers(_glfw_window);
}

void WindowsWindow::set_vsync(bool enabled) {
    if (enabled) {
        glfwSwapInterval(1);
    } else {
        glfwSwapBuffers(0);
    }
    _window_data.vsync = enabled;
}

bool WindowsWindow::is_vsync() const {
    return _window_data.vsync;
}
