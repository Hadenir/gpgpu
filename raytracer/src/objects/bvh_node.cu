#include "bvh_node.cuh"

namespace obj
{
    __device__ BvhNode::~BvhNode()
    {
        delete _left;
        delete _right;
    }

    __device__ bool BvhNode::hit(const math::Ray& ray, float t_min, float t_max, HitResult& result) const
    {
        if(!_aabb.hit(ray, t_min, t_max)) return false;

        bool hit_left = _left->hit(ray, t_min, t_max, result);
        bool hit_right = _right->hit(ray, t_min, hit_left ? result.t : t_max, result);

        return hit_left || hit_right;
    }

    __device__ bool BvhNode::bounding_box(math::AABB& result) const
    {
        result = _aabb;
        return true;
    }


    __device__ BvhNode::BvhNode(RenderObject** objects, int start, int end, curandState_t* curandState)
    {
        float rnum = curand_uniform(curandState);
        int axis = rnum < 0.3 ? 0
                 : rnum < 0.6 ? 1
                 : 2;

        size_t span = end - start;
        if(span < 2)
        {
            _left = objects[start];
            _right = objects[start];
        }
        else if(span == 2)
        {
            if(compare(objects[start], objects[start+1], axis))
            {
                _left = objects[start];
                _right = objects[start+1];
            }
            else
            {
                _left = objects[start+1];
                _right = objects[start];
            }
        }
        else
        {
            sort(objects, start, end, axis);

            auto mid = start + span / 2;
            _left = new BvhNode(objects, start, mid, curandState);
            _right = new BvhNode(objects, mid, end, curandState);
        }

        math::AABB box_left, box_right;
        if(!_left->bounding_box(box_left) || !_right->bounding_box(box_right))
        {}
        _aabb = surrounding_box(box_left, box_right);
    }

    __device__ void BvhNode::sort(RenderObject** objects, int start, int end, int axis)
    {
        if(start == end) return;

        for(int i = start + 1; i < end; i++)
        {
            RenderObject* x = objects[i];
            int j = i - 1;
            while(j >= start && compare(objects[j], x, axis))
            {
                objects[j+1] = objects[j];
                j--;
            }
            objects[j+1] = x;
        }
    }

    __device__ bool BvhNode::compare(RenderObject* a, RenderObject* b, int axis)
    {
        math::AABB box_a, box_b;
        a->bounding_box(box_a);
        b->bounding_box(box_b);

        return box_a.minimum()[axis] < box_b.minimum()[axis];
    }
}
