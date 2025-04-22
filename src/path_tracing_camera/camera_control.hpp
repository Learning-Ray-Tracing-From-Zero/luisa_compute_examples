#pragma once

#include "camera.hpp"


class OrbitController {
private:
    Camera& _camera;
    float _move_speed;
    float _rotate_speed;
    float _zoom_speed;

public:
    explicit OrbitController(
        Camera& camera,
        float move_speed,
        float rotate_speed,
        float zoom_speed
    ) noexcept
        : _camera { camera }
        , _move_speed { move_speed }
        , _rotate_speed { rotate_speed }
        , _zoom_speed { zoom_speed }
    {
        // make sure the camera is valid
        _camera.front = normalize(_camera.front);
        _camera.right = normalize(cross(_camera.front, _camera.up));
        _camera.up = normalize(cross(_camera.right, _camera.front));
        _camera.fov = std::clamp(_camera.fov, 1.0f, 179.0f);
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

    [[nodiscard]] auto move_speed() const noexcept { return _move_speed; }
    [[nodiscard]] auto rotate_speed() const noexcept { return _rotate_speed; }
    [[nodiscard]] auto zoom_speed() const noexcept { return _zoom_speed; }
    void set_move_speed(float speed) noexcept { _move_speed = speed; }
    void set_rotate_speed(float speed) noexcept { _rotate_speed = speed; }
    void set_zoom_speed(float speed) noexcept { _zoom_speed = speed; }
};
