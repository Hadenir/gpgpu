#pragma once

#include "render_object.cuh"

using namespace math;

namespace obj
{
    class Sphere : public RenderObject
    {
    public:
        __device__ Sphere() {}
        __device__ Sphere(Vec3 center, float radius) : _center(center), _radius(radius) {}

        __device__ inline Vec3& center() { return _center; }
        __device__ inline const Vec3& center() const { return _center; }
        __device__ inline float radius() {return _radius; }

        __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& result) const;

    private:
        Vec3 _center;
        float _radius;
    };
}
