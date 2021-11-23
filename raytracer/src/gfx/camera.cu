#include "camera.cuh"
#include "../utils.cuh"

using namespace math;

namespace gfx
{
    __host__ __device__ Camera::Camera(float distance, math::Vec3 look_at, float fov, float aspect_ratio)
        : _target(look_at), _aspect_ratio(aspect_ratio), _x_angle(0), _y_angle(0), _distance(distance)
    {
        _theta = to_radians(fov);
        update();
    }

    __host__ __device__ Ray Camera::calculate_ray(float u, float v) const
    {
        return Ray(_origin, _lower_left + u * _horizontal + v * _vertical - _origin);
    }

    __host__ void Camera::move(float dx, float dy)
    {
        _x_angle += dx;
        if(_x_angle > PI) _x_angle -= 2 * PI;
        if(_x_angle < -PI) _x_angle += 2 * PI;
        _y_angle += dy;
        if(_y_angle > PI / 2) _y_angle = PI / 2;
        if(_y_angle < -PI / 2) _y_angle = -PI / 2;

        update();
    }

    __host__ __device__ void Camera::update()
    {
        float h = tan(_theta / 2);
        float viewport_height = 2 * h;
        float viewport_width = _aspect_ratio * viewport_height;

        _origin = _target + Vec3(_distance * sin(_x_angle) * cos(_y_angle), _distance * sin(_y_angle), _distance * cos(_y_angle) * cos(_x_angle));

        Vec3 w = (_origin - _target).normalized();
        Vec3 u = Vec3::up().cross(w).normalized();
        Vec3 v = w.cross(u);

        _horizontal = viewport_width * u;
        _vertical = viewport_height * v;
        _lower_left = _origin - _horizontal / 2 - _vertical / 2 - w;
    }
}
