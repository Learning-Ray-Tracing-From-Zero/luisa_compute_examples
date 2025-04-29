#pragma once

#include <gui/event.hpp>

#include <cstdint>
#include <string>
#include <functional>

using namespace std::literals::string_literals;


struct WindowProps {
    std::string title;
    std::uint32_t width;
    std::uint32_t height;

    WindowProps(
        const std::string& title_ = "Path Tracing With DearImgui"s,
        std::uint32_t width_ = 1536,
        std::uint32_t height_ = 864
    )
        : title(title_)
        , width(width_)
        , height(height_)
    {}
};


class Window {
public:
    using EventCallback = std::function<void(Event&)>;

    static Window* create(const WindowProps& props = WindowProps {});
    virtual ~Window() = default;

    virtual void update() = 0;
    virtual std::uint32_t width() const = 0;
    virtual std::uint32_t height() const = 0;

    virtual void set_event_callback(const EventCallback& callback) = 0;
    virtual void set_vsync(bool enabled) = 0;
    virtual bool is_vsync() const = 0;
};
