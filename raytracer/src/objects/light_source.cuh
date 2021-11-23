#pragma once

#include "../math/vec3.cuh"

using namespace math;

namespace obj
{
    class LightSource
    {
    public:
        __device__ LightSource() {}
        __device__ LightSource(Vec3 position, Vec3 color) : _position(position), _color(color) {}

        __device__ inline Vec3& position() { return _position; }
        __device__ inline const Vec3& position() const { return _position; }
        __device__ inline Vec3& color() { return _color; }
        __device__ inline const Vec3& color() const { return _color; }

    private:
        Vec3 _position;
        Vec3 _color;
    };
}
