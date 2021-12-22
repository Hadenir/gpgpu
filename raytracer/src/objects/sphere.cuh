#pragma once

#include "render_object.cuh"

using namespace math;

namespace obj
{
    class Sphere : public RenderObject
    {
    public:
        Sphere() {}
        Sphere(Vec3 center, float radius, Vec3 color)
            : _center(center), _radius(radius), _color(color) {}

        inline Vec3& center() { return _center; }
        inline const Vec3& center() const { return _center; }
        inline float radius() {return _radius; }

        virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& result) const;

    private:
        Vec3 _center;
        float _radius;
        Vec3 _color;
    };
}
