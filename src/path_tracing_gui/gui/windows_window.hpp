#pragma once

#include <core/window.hpp>
// #include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>


class WindowsWindow : public Window{
private:
    GLFWwindow* _glfw_window;

    struct WindowData {
        std::string title;
        std::uint32_t width;
        std::uint32_t height;
        bool vsync;
        EventCallback event_callback;
    };
    WindowData _window_data;

public:
    WindowsWindow(const WindowProps& props);

    virtual ~WindowsWindow();

    void update() override;

    std::uint32_t width() const override {return _window_data.width;}
    std::uint32_t height() const override {return _window_data.height;}

    void set_event_callback(const EventCallback& callback) override {
        _window_data.event_callback = callback;
    }

    void set_vsync(bool enabled) override;
    bool is_vsync() const override;

private:
    virtual void init(const WindowProps& props);
    virtual void shutdown();
};
