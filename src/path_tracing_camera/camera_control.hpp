#pragma once

#include "camera.hpp"

#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>

#include <cmath>
#include <unordered_map>


class OrbitController {
private:
    Camera& _camera;
    GLFWwindow* _glfw_window;
    bool _is_moved;
    glm::dvec2 _cursor_pos;
    double _delta_time;
    float _move_speed;
    float _rotate_speed;
    float _zoom_speed;
    std::unordered_map<int, bool> key_state;

public:
    explicit OrbitController(
        Camera& camera,
        GLFWwindow* glfw_window,
        float move_speed,
        float rotate_speed,
        float zoom_speed
    ) noexcept
        : _camera { camera }
        , _glfw_window { glfw_window }
        , _is_moved { false }
        , _cursor_pos { 0.0, 0.0 }
        , _delta_time { 0.0 }
        , _move_speed { move_speed }
        , _rotate_speed { rotate_speed }
        , _zoom_speed { zoom_speed }
    {
        // make sure the camera is valid
        _camera.front = normalize(_camera.front);
        _camera.right = normalize(cross(_camera.front, _camera.up));
        _camera.up = normalize(cross(_camera.right, _camera.front));
        _camera.fov = std::clamp(_camera.fov, 1.0f, 179.0f);

        // Register callback functions
        glfwSetWindowUserPointer(_glfw_window, this);
        glfwSetKeyCallback(_glfw_window, key_callback);
        glfwSetMouseButtonCallback(_glfw_window, mouse_button_callback);
    }

    static void key_callback(
        GLFWwindow* glfw_window,
        int key,
        int scancode,
        int action,
        int mods
    ) {
        auto self = static_cast<OrbitController*>(glfwGetWindowUserPointer(glfw_window));
        if (self) { self->handle_key_callback(key, action); }
    }

    void handle_key_callback(int key, int action) {
        if (action == GLFW_PRESS) {
            key_state[key] = true;
            _is_moved = true;
        } else if (action == GLFW_RELEASE) {
            key_state[key] = false;
            _is_moved = false;
        }
    }

    static void mouse_button_callback(GLFWwindow* glfw_window, int button, int action, int mods) {
        auto self = static_cast<OrbitController*>(glfwGetWindowUserPointer(glfw_window));
        if (self) { self->handle_mouse_button_callback(glfw_window, button, action); }
    }

    void handle_mouse_button_callback(GLFWwindow* glfw_window, int button, int action) {
        if (action == GLFW_PRESS) {
            key_state[button] = true;
            auto pos_x { 0.0 };
            auto pos_y { 0.0 };
            glfwGetCursorPos(glfw_window, pos_x, pos_y);
            _cursor_pos = { pos_x, pos_y };
        } else if (action == GLFW_RELEASE) {
            key_state[button] = false;
            _is_moved = false;
        }
    }

    void handle_key() {
        if (!_is_moved) { return ; }
        auto dt = static_cast<float>(_delta_time / 1000.0);
        if (key_state[GLFW_KEY_W]) { rotate_pitch(dt); }
        if (key_state[GLFW_KEY_S]) { rotate_pitch(-dt); }
        if (key_state[GLFW_KEY_A]) { rotate_yaw(dt); }
        if (key_state[GLFW_KEY_D]) { rotate_yaw(-dt); }
        if (key_state[GLFW_KEY_Q]) { rotate_roll(-dt); }
        if (key_state[GLFW_KEY_E]) { rotate_roll(dt); }
        if (key_state[GLFW_KEY_MINUS]) { zoom(-dt); }
        if (key_state[GLFW_KEY_EQUAL]) { zoom(dt); }
        if (key_state[GLFW_KEY_UP]) {
            if (key_state[GLFW_KEY_LEFT_SHIFT]
                || key_state[GLFW_KEY_RIGHT_SHIFT]
            ) {
                move_forward(dt);
            } else {
                move_up(dt);
            }
        }
        if (key_state[GLFW_KEY_DOWN]) {
            if (
                key_state[GLFW_KEY_LEFT_SHIFT]
                || key_state[GLFW_KEY_RIGHT_SHIFT]
            ) {
                move_forward(-dt);
            } else {
                move_up(-dt);
            }
        }
        if (key_state[GLFW_KEY_LEFT]) { move_right(-dt); }
        if (key_state[GLFW_KEY_RIGHT]) { move_right(dt); }
    }

    void handle_cursor() {
        auto pos_x { 0.0 };
        auto pos_y { 0.0 };
        glfwGetCursorPos(glfw_window, pos_x, pos_y);
        auto cursor_pos_delta = glm::vec2(
             pos_x - _cursor_pos,
             pos_y - _cursor_pos
        );
        _cursor_pos = pos_x;
        _cursor_pos = pos_y;

        is_moved = (cursor_pos_delta.x || cursor_pos_delta.y) ? true : false;
        if (!_is_moved) { return ; }

        auto dt = static_cast<float>(_delta_time / 1000.0);
        if (cursor_pos_delta.x > 0) {
            rotate_yaw(-dt);
        } else {
            rotate_yaw(dt);
        }
        if (cursor_pos_delta.y > 0) {
            rotate_pitch(dt);
        } else {
            rotate_pitch(-dt);
        }
    }

    void zoom(float scale) noexcept {
        _camera.fov = std::clamp(
            _camera.fov * std::pow(2.0f, -scale * _zoom_speed),
            1.0f,
            179.0f
        );
    }

    void move_right(float dx) noexcept {
        _camera.position += _camera.right * dx * _move_speed;
    }

    void move_up(float dy) noexcept {
        _camera.position += _camera.up * dy * _move_speed;
    }

    void move_forward(float dz) noexcept {
        _camera.position += _camera.front * dz * _move_speed;
    }

    void rotate_roll(float angle) noexcept {
        auto m = make_float3x3(rotation(_camera.front, radians(_rotate_speed * angle)));
        _camera.up = normalize(m * _camera.up);
        _camera.right = normalize(m * _camera.right);
    }

    void rotate_yaw(float angle) noexcept {
        auto m = make_float3x3(rotation(_camera.up, radians(_rotate_speed * angle)));
        _camera.front = normalize(m * _camera.front);
        _camera.right = normalize(m * _camera.right);
    }

    void rotate_pitch(float angle) noexcept {
        auto m = make_float3x3(rotation(_camera.right, radians(_rotate_speed * angle)));
        _camera.front = normalize(m * _camera.front);
        _camera.up = normalize(m * _camera.up);
    }

    [[nodiscard]] auto delta_time() const noexcept { return _delta_time; }
    [[nodiscard]] auto is_moved() const noexcept { return _is_moved; }
    [[nodiscard]] auto move_speed() const noexcept { return _move_speed; }
    [[nodiscard]] auto rotate_speed() const noexcept { return _rotate_speed; }
    [[nodiscard]] auto zoom_speed() const noexcept { return _zoom_speed; }
    void set_delta_time(float delta_time) noexcept { _delta_time = delta_time; }
    void set_move_speed(float speed) noexcept { _move_speed = speed; }
    void set_rotate_speed(float speed) noexcept { _rotate_speed = speed; }
    void set_zoom_speed(float speed) noexcept { _zoom_speed = speed; }
};
