#pragma once

#include <gui/event.hpp>

#include <cstdint>
#include <sstream>


class WindowResizeEvent : public Event {
private:
    std::uint32_t _width;
    std::uint32_t _height;

public:
    WindowResizeEvent(std::uint32_t width, std::uint32_t height)
        : _width(width)
        , _height(height)
    {}

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)

    std::uint32_t width() const { return _width; }
    std::uint32_t height() const { return _height; }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << _width << ", " << _height;
        return ss.str();
    }
};


class WindowCloseEvent : public Event {
public:
    WindowCloseEvent() = default;

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};


class AppTickEvent : public Event {
public:
    AppTickEvent() = default;

    EVENT_CLASS_TYPE(AppTick)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};


class AppRenderEvent : public Event {
public:
    AppRenderEvent() = default;

    EVENT_CLASS_TYPE(AppRender)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
};
