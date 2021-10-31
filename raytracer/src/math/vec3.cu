#include <cmath>
#include <algorithm>

#include "vec3.cuh"

namespace math
{
    Vec3 Vec3::operator+(const Vec3& other) const
    {
        return Vec3(_v[0] + other._v[0], _v[1] + other._v[1], _v[1] + other._v[1]);
    }

    Vec3 Vec3::operator-(const Vec3& other) const
    {
        return Vec3(_v[0] - other._v[0], _v[1] - other._v[1], _v[1] - other._v[1]);
    }

    Vec3& Vec3::operator+=(const Vec3& other)
    {
        _v[0] += other._v[0];
        _v[1] += other._v[1];
        _v[2] += other._v[2];
        return *this;
    }

    Vec3& Vec3::operator-=(const Vec3& other)
    {
        _v[0] -= other._v[0];
        _v[1] -= other._v[1];
        _v[2] -= other._v[2];
        return *this;
    }

    Vec3& Vec3::operator*=(float coeff)
    {
        _v[0] *= coeff;
        _v[1] *= coeff;
        _v[2] *= coeff;
        return *this;
    }

    Vec3& Vec3::operator/=(float coeff)
    {
        _v[0] /= coeff;
        _v[1] /= coeff;
        _v[2] /= coeff;
        return *this;
    }

    bool Vec3::operator==(const Vec3& other) const
    {
        return fabs(_v[0] - other._v[0]) <= EPSILON
            && fabs(_v[1] - other._v[1]) <= EPSILON
            && fabs(_v[2] - other._v[2]) <= EPSILON;
    }

    bool Vec3::operator!=(const Vec3& other) const
    {
        return fabs(_v[0] - other._v[0]) > EPSILON
            || fabs(_v[1] - other._v[1]) > EPSILON
            || fabs(_v[2] - other._v[2]) > EPSILON;
    }

    float Vec3::dot(const Vec3& other) const
    {
        return _v[0] * other._v[0] + _v[1] * other._v[1] + _v[2] * other._v[2];
    }

    Vec3 Vec3::cross(const Vec3& other) const
    {
        return Vec3(
            _v[1] * other._v[2] - _v[2] * other._v[1],
            _v[2] * other._v[0] - _v[0] * other._v[2],
            _v[0] * other._v[1] - _v[1] * other._v[0]
        );
    }

    float Vec3::length() const
    {
        return sqrtf(length_sq());
    }

    float Vec3::length_sq() const
    {
        return _v[0] * _v[0] + _v[1] * _v[1] + _v[2] * _v[2];
    }

    void Vec3::normalize()
    {
        auto vec = absolute();
        float m = max(vec._v[0], max(vec._v[1], vec._v[2]));

        if(m > 0)
        {
            float len = length();
            _v[0] /= len;
            _v[1] /= len;
            _v[2] /= len;
        }
    }

    Vec3 Vec3::absolute() const
    {
        return Vec3(fabs(_v[0]), fabs(_v[1]), fabs(_v[2]));
    }

    Vec3 operator*(float coeff, const Vec3& vec)
    {
        return Vec3(coeff * vec._v[0], coeff * vec._v[1], coeff * vec._v[2]);
    }

    Vec3 operator/(float coeff, const Vec3& vec)
    {
        return Vec3(coeff / vec._v[0], coeff / vec._v[1], coeff / vec._v[2]);
    }
}
