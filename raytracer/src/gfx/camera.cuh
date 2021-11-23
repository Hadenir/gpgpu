#pragma once

#include "../math/vec3.cuh"
#include "../math/ray.cuh"

namespace gfx
{
    class Camera
    {
    public:
        __host__ __device__ Camera(float distance, math::Vec3 look_at, float fov, float aspect_ratio);

        __host__ __device__ math::Ray calculate_ray(float u, float v) const;

        __host__ __device__ void move(float dx, float dy);

    private:
        math::Vec3 _origin;
        math::Vec3 _target;
        float _x_angle;
        float _y_angle;
        float _distance;

        float _theta;
        float _aspect_ratio;

        math::Vec3 _lower_left;
        math::Vec3 _horizontal;
        math::Vec3 _vertical;

        __host__ __device__ void update();
    };
}
