#pragma once

namespace math
{
    class Vec3
    {
    public:
        Vec3() = default;
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

        __host__ __device__ inline Vec3 operator+(const Vec3& other) const;
        __host__ __device__ inline Vec3 operator-(const Vec3& other) const;
        __host__ __device__ inline Vec3& operator+=(const Vec3& other);
        __host__ __device__ inline Vec3& operator-=(const Vec3& other);
        __host__ __device__ inline Vec3& operator*=(float coeff);
        __host__ __device__ inline Vec3& operator/=(float coeff);
        __host__ __device__ inline bool operator==(const Vec3& other) const;
        __host__ __device__ inline bool operator!=(const Vec3& other) const;

        __host__ __device__ inline friend Vec3 operator*(float coeff, const Vec3& vec);
        __host__ __device__ inline friend Vec3 operator/(float coeff, const Vec3& vec);

        __host__ __device__ inline float dot(const Vec3& other) const;
        __host__ __device__ inline Vec3 cross(const Vec3& other) const;

        __host__ __device__ inline float length() const;
        __host__ __device__ inline float length_sq() const;
        __host__ __device__ inline void normalize();

    private:
        const float EPSILON = FLT_EPSILON;
        float _v[3];

        __host__ __device__ inline Vec3 absolute() const;
    };
}
