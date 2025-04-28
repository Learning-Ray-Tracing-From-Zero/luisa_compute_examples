#include <core/application.hpp>


int main(int argc, char** argv) {
    auto app = CreateApplication();
    app->run();
    delete app;

    return 0;
}
