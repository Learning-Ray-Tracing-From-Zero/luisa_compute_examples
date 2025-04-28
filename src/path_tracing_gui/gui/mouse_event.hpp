#pragma once

#include <gui/event.hpp>
#include <gui/mouse_codes.hpp>

#include <sstream>


class MouseMovedEvent : public Event {
private:
    float _mouse_pos_x;
    float _mouse_pos_y;

public:
    MouseMovedEvent()
        : _mouse_pos_x { 0.0f }
        , _mouse_pos_y { 0.0f }
    {}

    MouseMovedEvent(const float x, const float y)
        : _mouse_pos_x { x }
        , _mouse_pos_y { y }
    {}

    float mouse_pos_x() const { return _mouse_pos_x; }
    float mouse_pos_y() const { return _mouse_pos_y; }

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "MouseMovedEvent: " << _mouse_pos_x << ", " << _mouse_pos_y;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseMoved)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
};


class MouseScrolledEvent : public Event {
private:
    float _offset_x;
    float _offset_y;

public:
    MouseScrolledEvent(const float xOffset, const float yOffset)
        : _offset_x(xOffset)
        , _offset_y(yOffset)
    {}

    float offset_x() const { return _offset_x; }
    float offset_y() const { return _offset_y; }

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "MouseScrolledEvent: " << offset_x() << ", " << offset_y();
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseScrolled)
    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

};


class MouseButtonEvent : public Event {
protected:
    MouseCode _mouse_button;

protected:
    MouseButtonEvent(const MouseCode button)
        : _mouse_button(button)
    {}

public:
    MouseCode mouse_button() const { return _mouse_button; }

    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput | EventCategoryMouseButton)
};


class MouseButtonPressedEvent : public MouseButtonEvent {
public:
    MouseButtonPressedEvent(const MouseCode button)
        : MouseButtonEvent(button)
    {}

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "MouseButtonPressedEvent: " << _mouse_button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonPressed)
};


class MouseButtonReleasedEvent : public MouseButtonEvent {
public:
    MouseButtonReleasedEvent(const MouseCode button)
        : MouseButtonEvent(button)
    {}

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "MouseButtonReleasedEvent: " << _mouse_button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonReleased)
};
