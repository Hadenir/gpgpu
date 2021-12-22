#pragma once

#include "render_object.cuh"

namespace obj
{
    class RenderList : public RenderObject
    {
    public:
        RenderList() {}
        RenderList(RenderObject** objects, size_t n) : _objects(objects), _size(n) {}

        virtual bool hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const;

    private:
        RenderObject** _objects = nullptr;
        size_t _size = 0;
    };
}
