#pragma once

#include "render_object.cuh"
#include "render_list.cuh"

#include <curand.h>
#include <curand_kernel.h>

namespace obj
{
    class BvhNode : public RenderObject
    {
    public:
        __device__ BvhNode() {}
        __device__ BvhNode(const RenderList& list, curandState_t* curandState) : BvhNode(list._objects, 0, list._size, curandState) {}

        __device__ bool hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const override;
        __device__ bool bounding_box(math::AABB& result) const override;

    private:
        RenderObject *_left, *_right;
        math::AABB _aabb;

        __device__ BvhNode(RenderObject** objects, size_t start, size_t end, curandState_t* curandState);

        __device__ void sort(RenderObject** objects, size_t start, size_t end, int axis);

        __device__ bool compare(RenderObject* a, RenderObject* b, int axis);
    };
}
