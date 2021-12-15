#pragma once

#include "render_object.cuh"

using namespace math;

namespace obj
{
    class Sphere : public RenderObject
    {
    public:
        __device__ Sphere() {}
        __device__ Sphere(Vec3 center, float radius, Vec3 color)
            : _center(center), _radius(radius), _color(color) {}

        __device__ inline Vec3& center() { return _center; }
        __device__ inline const Vec3& center() const { return _center; }
        __device__ inline float radius() {return _radius; }

        __device__ bool hit(const Ray& ray, float t_min, float t_max, HitResult& result) const override;
        __device__ bool bounding_box(AABB& result) const override;

    private:
        Vec3 _center;
        float _radius;
        Vec3 _color;
    };
}
