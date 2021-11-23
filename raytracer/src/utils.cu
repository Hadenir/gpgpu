#include "utils.cuh"

void cuda_check(cudaError_t result, const char* func, const char* file, const int line)
{
    if(result == cudaSuccess) return;

    cudaDeviceReset();

    std::stringstream ss;
    ss << "CUDA error " << (unsigned int)result << " at " << file << ':' << line
        << " in '" << func << "': " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
}

__host__ __device__ float to_radians(float degrees)
{
    return degrees / 180.f * PI;
}

__host__ __device__ float to_degrees(float radians)
{
    return radians / PI * 180.0f;
}
