#include "render_object.cuh"

namespace obj
{
    __device__ math::AABB RenderObject::surrounding_box(const math::AABB& box1, const math::AABB& box2) const
    {
        math::Vec3 minimum(
            fmin(box1.minimum().x(), box2.minimum().x()),
            fmin(box1.minimum().y(), box2.minimum().y()),
            fmin(box1.minimum().z(), box2.minimum().z())
        );
        math::Vec3 maximum(
            fmax(box1.maximum().x(), box2.maximum().x()),
            fmax(box1.maximum().y(), box2.maximum().y()),
            fmax(box1.maximum().z(), box2.maximum().z())
        );

        return math::AABB(minimum, maximum);
    }
}
