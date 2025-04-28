#pragma once

#include <core/window.hpp>
#include <gui/event.hpp>
#include <gui/application_event.hpp>

#include <memory>


class Application {
private:
    std::unique_ptr<Window> _window;
    bool running { true };

public:
    Application();
    virtual ~Application();

    void run();
    void event(Event& event);

private:
    bool window_close(WindowCloseEvent& event);
};

Application* CreateApplication();
