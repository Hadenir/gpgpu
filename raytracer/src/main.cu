#include <iostream>
#include <cstdint>

#include "utils.cuh"
#include "gfx/display.cuh"
#include "gfx/renderer.cuh"
#include "gfx/camera.cuh"
#include "math/vec3.cuh"
#include "math/ray.cuh"
#include "objects/render_list.cuh"
#include "objects/sphere.cuh"

using namespace gfx;
using namespace math;
using namespace obj;
typedef unsigned int uint;

__device__ Vec3 calculate_color(const Ray& ray, RenderObject* world)
{
    HitResult result;
    if(world->hit(ray, 0.0f, FLT_MAX, result))
    {
        return 0.5f * (result.normal + Vec3::one());
    }
    else
    {
        const Vec3& direction = ray.direction();
        float t = 0.5f * (direction.y() + 1.0f);
        return (1.0f - t) * Vec3::one() + t * Vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void create_world(RenderObject** world)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        RenderObject** objects = new RenderObject*[3];
        objects[0] = new Sphere(Vec3(1.0f, 0.0f, -1.0f), 0.5f);
        objects[1] = new Sphere(Vec3(-1.0f, 0.0f, -1.0f), 0.5f);
        objects[2] = new Sphere(Vec3(0, -100.5f, -1.0f), 100.0f);

        *world = new RenderList(objects, 3);
    }
}

__global__ void render(RenderObject** world, Camera camera, int width, int height, float4* pixels)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);
    Ray ray = camera.calculate_ray(u, v);

    Vec3 col = calculate_color(ray, *world);

    pixels[x + y * width].x = col.r();
    pixels[x + y * width].y = col.g();
    pixels[x + y * width].z = col.b();
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
    int window_width = 1400;
    int window_height = 700;
    int resolution_width = window_width;
    int resolution_height = window_height;
    std::string window_title = "CUDA Raytracer - Konrad Brzozka - Procesory Graficzne w Zastosowaniach Obliczeniowych";

    Display display(window_title, window_width, window_height);
    Renderer renderer(resolution_width, resolution_height);

    dim3 block_size(32, 32);
    dim3 grid_size = calculate_grid_size(resolution_width, resolution_height, block_size);

    RenderObject** world;
    CUDA_CHECK(cudaMalloc(&world, sizeof(RenderObject*)));
    create_world<<<1, 1>>>(world);
    CUDA_CHECK(cudaDeviceSynchronize());

    Camera camera(3, Vec3(0, 0, -1), 90, (float)resolution_width / resolution_height);
    float mouse_x = 0, mouse_y = 0;
    while(!display.should_close())
    {
        float new_mouse_x, new_mouse_y;
        display.get_cursor_pos(new_mouse_x, new_mouse_y);

        float dx = new_mouse_x - mouse_x;
        float dy = new_mouse_y - mouse_y;
        mouse_x = new_mouse_x;
        mouse_y = new_mouse_y;

        if(display.is_dragging())
            camera.move(-dx / 100, dy / 100);

        renderer.clear();

        float4* framebuffer = renderer.get_framebuffer();
        render<<<grid_size, block_size>>>(world, camera, resolution_width, resolution_height, framebuffer);
        CUDA_CHECK(cudaDeviceSynchronize());

        renderer.blit();
        renderer.draw();
        display.show();
    }

    return 0;
}
