#pragma once

#include <gui/event.hpp>
#include <gui/key_codes.hpp>

#include <sstream>


class KeyEvent : public Event {
protected:
    KeyCode _key_code;

protected:
    KeyEvent(const KeyCode keycode)
        : _key_code(keycode)
    {}

public:
    KeyCode key_code() const { return _key_code; }

    EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
};


class KeyPressedEvent : public KeyEvent {
private:
    bool _is_repeat;

public:
    KeyPressedEvent(const KeyCode keycode, bool is_repeat_ = false)
        : KeyEvent(keycode)
        , _is_repeat { is_repeat_ }
    {}

    bool is_repeat() const { return _is_repeat; }

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "KeyPressedEvent: " << _key_code << " (repeat = " << _is_repeat << ")";
        return ss.str();
    }

    EVENT_CLASS_TYPE(KeyPressed)
};


class KeyReleasedEvent : public KeyEvent {
public:
    KeyReleasedEvent(const KeyCode keycode)
        : KeyEvent(keycode)
    {}

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "KeyReleasedEvent: " << _key_code;
        return ss.str();
    }

    EVENT_CLASS_TYPE(KeyReleased)
};


class KeyTypedEvent : public KeyEvent {
public:
    KeyTypedEvent(const KeyCode keycode)
        : KeyEvent(keycode)
    {}

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "KeyTypedEvent: " << _key_code;
        return ss.str();
    }

    EVENT_CLASS_TYPE(KeyTyped)
};
