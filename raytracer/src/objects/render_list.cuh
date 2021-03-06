#pragma once

#include "render_object.cuh"

namespace obj
{
    class RenderList : public RenderObject
    {
    public:
        __device__ RenderList() {}
        __device__ RenderList(RenderObject** objects, size_t n) : _objects(objects), _size(n) {}

        __device__ bool hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const override;
        __device__ bool bounding_box(math::AABB& result) const override;

    private:
        RenderObject** _objects = nullptr;
        size_t _size = 0;

        friend class BvhNode;
    };
}
