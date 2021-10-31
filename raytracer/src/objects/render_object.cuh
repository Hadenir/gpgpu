#pragma once

#include "../math/ray.cuh"

namespace obj
{
    struct HitResult
    {
        float t;
        math::Vec3 hit_point;
        math::Vec3 normal;
    };

    class RenderObject
    {
    public:
        __device__ virtual bool hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const = 0;
    };
}
