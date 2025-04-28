#pragma once

#include <cstdint>
#include <functional>
#include <iostream>


enum class EventType : std::uint8_t {
    None = 0,

    WindowClose,
    WindowResize,
    WindowFocus,
    WindowLostFocus,
    WindowMoved,

    AppTick,
    AppUpdate,
    AppRender,

    KeyPressed,
    KeyReleased,
    KeyTyped,

    MouseButtonPressed,
    MouseButtonReleased,
    MouseMoved,
    MouseScrolled
};


enum EventCategory : std::uint8_t {
    None = 0,
    EventCategoryApplication = 1 << 0,
    EventCategoryInput       = 1 << 1,
    EventCategoryKeyboard    = 1 << 2,
    EventCategoryMouse       = 1 << 3,
    EventCategoryMouseButton = 1 << 4
};


#define EVENT_CLASS_TYPE(type) \
    static EventType static_type() { return EventType::type; } \
    virtual EventType event_type() const override { return static_type(); } \
    virtual const char* name() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) \
    virtual int category_flags() const override { return category; }


class Event {
public:
    bool handled { false };

public:
    virtual ~Event() = default;

    virtual EventType event_type() const = 0;
    virtual const char* name() const = 0;
    virtual int category_flags() const = 0;
    virtual std::string to_string() const { return name(); }

    bool is_in_category(EventCategory category) {
        return category_flags() & category;
    }
};


class EventDispatcher {
private:
    Event& _event;

public:
    EventDispatcher(Event& event)
        : _event(event)
    {}

    // F will be deduced by the compiler
    template<typename T, typename F>
    auto dispatch(const F& func) {
        if (_event.event_type() == T::static_type()) {
            _event.handled |= func(static_cast<T&>(_event));
            return true;
        }

        return false;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Event& e) {
    return os << e.to_string();
}
