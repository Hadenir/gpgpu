#include "sphere.cuh"

namespace obj
{
    __device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitResult& result) const
    {
        Vec3 oc = ray.origin() - _center;
        float a = ray.direction().length_sq();
        float half_b = oc.dot(ray.direction());
        float c = oc.length_sq() - _radius * _radius;

        float discriminant = half_b * half_b - a * c;
        if(discriminant > 0)
        {
            float t = (-half_b - sqrtf(discriminant) / a);
            Vec3 hit_point = ray.point_at(t);
            if(t >= t_min && t <= t_max)
            {
                result.t = t;
                result.hit_point = hit_point;
                result.normal = (hit_point - _center) / _radius;
                result.color = _color;
                return true;
            }

            t = (-half_b + sqrtf(discriminant) / a);
            hit_point = ray.point_at(t);
            if(t >= t_min && t <= t_max)
            {
                result.t = t;
                result.hit_point = hit_point;
                result.normal = (hit_point - _center) / _radius;
                result.color = _color;
                return true;
            }
        }

        return false;
    }

    __device__ bool Sphere::bounding_box(AABB& result) const
    {
        result = AABB(
            _center - Vec3(_radius, _radius, _radius),
            _center + Vec3(_radius, _radius, _radius)
        );
        return true;
    }
}
