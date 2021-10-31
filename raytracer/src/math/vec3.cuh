#pragma once

#include <cfloat>

namespace math
{
    class Vec3
    {
    public:
        __host__ __device__ Vec3() {}
        __host__ __device__ Vec3(float v0, float v1, float v2) { _v[0] = v0, _v[1] = v1, _v[2] = v2; }

        __host__ __device__ inline float x() const { return _v[0]; }
        __host__ __device__ inline float y() const { return _v[1]; }
        __host__ __device__ inline float z() const { return _v[2]; }

        __host__ __device__ inline float r() const { return _v[0]; }
        __host__ __device__ inline float g() const { return _v[1]; }
        __host__ __device__ inline float b() const { return _v[2]; }

        __host__ __device__ inline const Vec3& operator+() const { return *this; }
        __host__ __device__ inline Vec3 operator-() const { return Vec3(-_v[0], -_v[1], -_v[2]); }
        __host__ __device__ inline float operator[](size_t i) const { return _v[i]; }
        __host__ __device__ inline float& operator[](size_t i) { return _v[i]; }

        __host__ __device__ Vec3& operator=(const Vec3& other);

        __host__ __device__ Vec3 operator+(const Vec3& other) const;
        __host__ __device__ Vec3 operator-(const Vec3& other) const;
        __host__ __device__ Vec3& operator+=(const Vec3& other);
        __host__ __device__ Vec3& operator-=(const Vec3& other);
        __host__ __device__ Vec3& operator*=(float coeff);
        __host__ __device__ Vec3& operator/=(float coeff);
        __host__ __device__ bool operator==(const Vec3& other) const;
        __host__ __device__ bool operator!=(const Vec3& other) const;

        __host__ __device__ float dot(const Vec3& other) const;
        __host__ __device__ Vec3 cross(const Vec3& other) const;

        __host__ __device__ float length() const;
        __host__ __device__ float length_sq() const;
        __host__ __device__ void normalize();
        __host__ __device__ Vec3 normalized() const;

        __host__ __device__ inline static Vec3 zero() { return Vec3(0.0f, 0.0f, 0.0f); }
        __host__ __device__ inline static Vec3 one() { return Vec3(1.0f, 1.0f, 1.0f); }

    private:
        const float EPSILON = FLT_EPSILON;
        float _v[3];

        __host__ __device__ friend Vec3 operator*(float coeff, const Vec3& vec);
        __host__ __device__ friend Vec3 operator/(const Vec3& vec, float coeff);
    };

    Vec3 operator*(float coeff, const Vec3& vec);
    Vec3 operator/(const Vec3& vec, float coeff);
}
