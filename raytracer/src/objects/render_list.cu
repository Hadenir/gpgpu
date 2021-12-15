#include "render_list.cuh"

#include "../math/vec3.cuh"

namespace obj
{
    __device__ bool RenderList::hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const
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

    __device__ bool RenderList::bounding_box(math::AABB& result) const
    {
        if(_objects == nullptr || _size < 1) return false;

        math::AABB tmp;
        bool first_box = true;

        for(int i = 0; i < _size; i++)
        {
            RenderObject* object = _objects[i];
            if(!object->bounding_box(tmp)) return false;

            result = first_box ? tmp : surrounding_box(result, tmp);
            first_box = false;
        }
    }
}
