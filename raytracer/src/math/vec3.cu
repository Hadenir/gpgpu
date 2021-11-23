#include <cmath>
#include <algorithm>

#include "vec3.cuh"

namespace math
{
    __host__ __device__ Vec3& Vec3::operator=(const Vec3& other)
    {
        _v[0] = other._v[0];
        _v[1] = other._v[1];
        _v[2] = other._v[2];
        return *this;
    }

    __host__ __device__ Vec3 Vec3::operator+(const Vec3& other) const
    {
        return Vec3(_v[0] + other._v[0], _v[1] + other._v[1], _v[2] + other._v[2]);
    }

    __host__ __device__ Vec3 Vec3::operator-(const Vec3& other) const
    {
        return Vec3(_v[0] - other._v[0], _v[1] - other._v[1], _v[2] - other._v[2]);
    }

    __host__ __device__ Vec3 Vec3::operator*(const Vec3& other) const
    {
        return Vec3(_v[0] * other._v[0], _v[1] * other._v[1], _v[2] * other._v[2]);
    }

    __host__ __device__ Vec3& Vec3::operator+=(const Vec3& other)
    {
        _v[0] += other._v[0];
        _v[1] += other._v[1];
        _v[2] += other._v[2];
        return *this;
    }

    __host__ __device__ Vec3& Vec3::operator-=(const Vec3& other)
    {
        _v[0] -= other._v[0];
        _v[1] -= other._v[1];
        _v[2] -= other._v[2];
        return *this;
    }

    __host__ __device__ Vec3& Vec3::operator*=(const Vec3& other)
    {
        _v[0] *= other._v[0];
        _v[1] *= other._v[1];
        _v[2] *= other._v[2];
        return *this;
    }

    __host__ __device__ Vec3& Vec3::operator*=(float coeff)
    {
        _v[0] *= coeff;
        _v[1] *= coeff;
        _v[2] *= coeff;
        return *this;
    }

    __host__ __device__ Vec3& Vec3::operator/=(float coeff)
    {
        _v[0] /= coeff;
        _v[1] /= coeff;
        _v[2] /= coeff;
        return *this;
    }

    __host__ __device__ bool Vec3::operator==(const Vec3& other) const
    {
        return fabs(_v[0] - other._v[0]) <= EPSILON
            && fabs(_v[1] - other._v[1]) <= EPSILON
            && fabs(_v[2] - other._v[2]) <= EPSILON;
    }

    __host__ __device__ bool Vec3::operator!=(const Vec3& other) const
    {
        return fabs(_v[0] - other._v[0]) > EPSILON
            || fabs(_v[1] - other._v[1]) > EPSILON
            || fabs(_v[2] - other._v[2]) > EPSILON;
    }

    __host__ __device__ float Vec3::dot(const Vec3& other) const
    {
        return _v[0] * other._v[0] + _v[1] * other._v[1] + _v[2] * other._v[2];
    }

    __host__ __device__ Vec3 Vec3::cross(const Vec3& other) const
    {
        return Vec3(
            _v[1] * other._v[2] - _v[2] * other._v[1],
            _v[2] * other._v[0] - _v[0] * other._v[2],
            _v[0] * other._v[1] - _v[1] * other._v[0]
        );
    }

    __host__ __device__ float Vec3::length() const
    {
        return sqrtf(length_sq());
    }

    __host__ __device__ float Vec3::length_sq() const
    {
        return _v[0] * _v[0] + _v[1] * _v[1] + _v[2] * _v[2];
    }

    __host__ __device__ void Vec3::normalize()
    {
        float len = length();
        if(len == 0) return;

        _v[0] /= len;
        _v[1] /= len;
        _v[2] /= len;
    }

    __host__ __device__ Vec3 Vec3::normalized() const
    {
        float len = length();
        if(len == 0) return *this;

        return Vec3(_v[0] / len, _v[1] / len, _v[2] / len);
    }

    __host__ __device__ Vec3& Vec3::clamp()
    {
        for(int i = 0; i < 3; i++)
        {
            if(_v[i] < 0) _v[i] = 0.0f;
            if(_v[i] > 1) _v[i] = 1.0f;
        }

        return *this;
    }

    __host__ __device__ Vec3 operator*(float coeff, const Vec3& vec)
    {
        return Vec3(coeff * vec._v[0], coeff * vec._v[1], coeff * vec._v[2]);
    }

    __host__ __device__ Vec3 operator/(const Vec3& vec, float coeff)
    {
        return Vec3(vec._v[0] / coeff, vec._v[1] / coeff, vec._v[2] / coeff);
    }
}
