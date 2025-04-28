#include <core/application.hpp>
#include <gui/application_event.hpp>

#include <functional>


#define BIND_EVENT_FN(x) std::bind(&Application::x, this, std::placeholders::_1)


Application::Application()
    : _window(std::unique_ptr<Window>(Window::create()))
{
    _window->set_event_callback(BIND_EVENT_FN(event));
}

Application::~Application() {}

void Application::event(Event &event) {
    EventDispatcher dispatcher(event);
    dispatcher.dispatch<WindowCloseEvent>(BIND_EVENT_FN(window_close));
    std::cout << event.to_string() << std::endl;
}

bool Application::window_close(WindowCloseEvent &event) {
    running = false;
    return true;
}


void Application::run() {
    WindowResizeEvent window_resize_event(1536, 864);
    if (window_resize_event.is_in_category(EventCategoryApplication)) {
        std::cout << window_resize_event.to_string() << std::endl;
    }
    if (window_resize_event.is_in_category(EventCategoryInput)) {
        std::cout << "WindowResizeEvent is in EventCategoryInput" << std::endl;
    }

    while (running) {
        _window->update();
    }
}


Application* CreateApplication() {
    return new Application();
}
