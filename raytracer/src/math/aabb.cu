#include "aabb.cuh"

namespace math
{
    __device__ bool AABB::hit(const Ray& ray, float t_min, float t_max) const
    {
        for(int i = 0; i < 3; i++)
        {
            float invD = 1.0f / ray.direction()[i];
            float t0 = (_minimum[i] - ray.origin()[i]) * invD;
            float t1 = (_maximum[i] - ray.origin()[i]) * invD;
            if(invD < 0.0f)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if(t_max <= t_min) return false;
        }

        return true;
    }
}
