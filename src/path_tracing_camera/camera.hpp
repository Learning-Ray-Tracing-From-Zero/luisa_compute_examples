#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>


struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};


struct Camera {
    float3 position;
    float3 front;
    float3 up;
    float3 right;
    float fov;
};

LUISA_STRUCT(Camera, position, front, up, right, fov) {
    // p: normalized pixel coordinate
    [[nodiscard]] auto generate_ray(Expr<float2> p) const noexcept {
        auto fov_radians = radians(fov);
        auto wi_local = make_float3(p * tan(0.5f * fov_radians), -1.0f);
        auto wi_world = normalize(wi_local.x * right + wi_local.y * up - wi_local.z * front);

        return make_ray(position, wi_world);
    }
};
