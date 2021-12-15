#pragma once

#include "../math/ray.cuh"
#include "../math/aabb.cuh"

namespace obj
{
    struct HitResult
    {
        float t;
        math::Vec3 hit_point;
        math::Vec3 normal;
        math::Vec3 color;
    };

    class RenderObject
    {
    public:
        __device__ virtual ~RenderObject() {}

        __device__ virtual bool hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const = 0;

        __device__ virtual bool bounding_box(math::AABB& result) const = 0;

    protected:
        __device__ math::AABB surrounding_box(const math::AABB& box1, const math::AABB& box2) const;
    };
}
