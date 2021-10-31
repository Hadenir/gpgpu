#include <iostream>
#include <cstdint>

#include "gfx/display.cuh"
#include "gfx/renderer.cuh"
#include "math/vec3.cuh"
#include "cuda_helpers.cuh"

#include <cuda_gl_interop.h>

typedef unsigned int uint;

__global__ void render(int width, int height, float4* pixels)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    pixels[x + y * width].x = x / float(width);
    pixels[x + y * width].y = y / float(height);
    pixels[x + y * width].z = 0;
}

dim3 calculate_grid_size(int width, int height, dim3 block_size)
{
    return dim3(
        width / block_size.x + (width % block_size.x == 0 ? 0 : 1),
        height / block_size.y + (height % block_size.y == 0 ? 0 : 1)
    );
}

int main(int argc, char* argv[])
{
    int width = 800;
    int height = 600;
    std::string window_title = "CUDA Raytracer - Konrad Brzozka - Procesory Graficzne w Zastosowaniach Obliczeniowych";

    gfx::Display display(window_title, width, height);
    gfx::Renderer renderer(width, height);

    int num_texels = width * height;
    int num_bytes = num_texels * sizeof(float4);
    float4* pixels;
    CUDA_CHECK(cudaMalloc(&pixels, num_bytes));

    dim3 block_size(32, 32);
    dim3 grid_size = calculate_grid_size(width, height, block_size);

    while(!display.should_close())
    {
        renderer.clear();

        render<<<grid_size, block_size>>>(width, height, pixels);
        CUDA_CHECK(cudaDeviceSynchronize());

        renderer.blit(pixels);
        renderer.draw();
        display.show();
    }

    return 0;
}
