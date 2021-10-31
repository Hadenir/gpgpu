#include "ray.cuh"

namespace math
{
    __host__ __device__ Ray::Ray(const Vec3& origin, const Vec3& direction)
        : _origin(origin), _direction(direction.normalized())
    {}

    __host__ __device__ Vec3 Ray::point_at(float t) const
    {
        return _origin + t * _direction;
    }
}
