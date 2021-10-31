#pragma once

#include "vec3.cuh"

namespace math
{
    class Ray
    {
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const Vec3& origin, const Vec3& direction);

        __host__ __device__ inline Vec3& origin() { return _origin; }
        __host__ __device__ inline const Vec3& origin() const { return _origin; }
        __host__ __device__ inline Vec3& direction() { return _direction; }
        __host__ __device__ inline const Vec3& direction() const { return _direction; }

        __host__ __device__ Vec3 point_at(float t) const;

    private:
        Vec3 _origin, _direction;
    };
}
