#pragma once

#include "vec3.cuh"
#include "ray.cuh"

namespace math
{
    class AABB
    {
    public:
        __host__ __device__ AABB() {}
        __host__ __device__ AABB(const Vec3& minimum, const Vec3& maximum) : _minimum(minimum), _maximum(maximum) {}

        __host__ __device__ inline Vec3& minimum() { return _minimum; }
        __host__ __device__ inline const Vec3& minimum() const { return _minimum; }
        __host__ __device__ inline Vec3& maximum() { return _maximum; }
        __host__ __device__ inline const Vec3& maximum() const { return _maximum; }

        __device__ bool hit(const Ray& ray, float t_min, float t_max) const;

    private:
        Vec3 _minimum, _maximum;
    };
}
