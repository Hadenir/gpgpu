#pragma once

#include "../math/vec3.cuh"

using namespace math;

namespace obj
{
    class LightSource
    {
    public:
        LightSource() {}
        LightSource(Vec3 position, Vec3 color) : _position(position), _color(color) {}

        inline Vec3& position() { return _position; }
        inline const Vec3& position() const { return _position; }
        inline Vec3& color() { return _color; }
        inline const Vec3& color() const { return _color; }

    private:
        Vec3 _position;
        Vec3 _color;
    };
}
