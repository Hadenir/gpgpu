#include "render_list.cuh"

namespace obj
{
    bool RenderList::hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const
    {
        HitResult temp_result;
        bool hit_anything = false;
        float closest = t_max;

        for(size_t i = 0; i < _size; i++)
        {
            if(_objects[i]->hit(ray, t_min, closest, temp_result))
            {
                hit_anything = true;
                result = temp_result;
                closest = result.t;
            }
        }

        return hit_anything;
    }
}
