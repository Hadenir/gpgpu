#pragma once

#include "../math/vec3.cuh"
#include "../math/ray.cuh"

using namespace math;

namespace gfx
{
    class Camera
    {
    public:
        __host__ __device__ Camera(float distance, Vec3 look_at, float fov, float aspect_ratio);

        __host__ __device__ inline const Vec3& position() const { return _origin; }

        __host__ __device__ Ray calculate_ray(float u, float v) const;

        __host__ __device__ void move(float dx, float dy);

    private:
        Vec3 _origin;
        Vec3 _target;
        float _x_angle;
        float _y_angle;
        float _distance;

        float _theta;
        float _aspect_ratio;

        Vec3 _lower_left;
        Vec3 _horizontal;
        Vec3 _vertical;

        __host__ __device__ void update();
    };
}
