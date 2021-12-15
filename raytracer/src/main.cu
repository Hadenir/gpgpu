#include <iostream>
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.cuh"
#include "gfx/display.cuh"
#include "gfx/renderer.cuh"
#include "gfx/camera.cuh"
#include "math/vec3.cuh"
#include "math/ray.cuh"
#include "objects/render_list.cuh"
#include "objects/sphere.cuh"
#include "objects/light_source.cuh"

using namespace gfx;
using namespace math;
using namespace obj;
typedef unsigned int uint;

__device__ const int LIGHTS_COUNT = 10;
__device__ const int SPHERES_COUNT = 1000;

__device__ Vec3 calculate_color(const Ray& ray, RenderList* world, LightSource* lights, const Vec3& camera_position)
{
    float k_s = 0.2f;
    float k_d = 0.9f;
    float k_a = 0.1f;
    float alpha = 100.0f;
    Vec3 ambient(0.1f, 0.1f, 0.1f);

    HitResult result;
    if(world->hit(ray, 0.0f, FLT_MAX, result))
    {
        LightSource& light = lights[0];
        auto N = result.normal;
        auto L_m = (light.position() - result.hit_point).normalized(); // direction from surface to light
        auto R_m = 2.0f * L_m.dot(N) * N - L_m; // direction of perfectly reflected ray
        auto V = (camera_position - result.hit_point).normalized(); // direction from surface to the camera

        auto light_color = k_a * ambient;
        auto diffuse_intensity = L_m.dot(N);
        if(diffuse_intensity > 0)
            light_color += k_d * diffuse_intensity * light.color();
        auto specular_intensity = R_m.dot(V);
        if(specular_intensity > 0)
            light_color += k_s * powf(specular_intensity, alpha) * light.color();

        auto color = (light_color * result.color).clamp();
        return color;
    }
    else
    {
        const Vec3& direction = ray.direction();
        float t = 0.5f * (direction.y() + 1.0f);
        return (1.0f - t) * Vec3::one() + t * Vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void init_curand(unsigned int seed, curandState_t* states)
{
    curand_init(seed, blockIdx.x, 0, states + blockIdx.x);
}

__global__ void create_objects(RenderObject** objects, LightSource* lights, RenderList* world, curandState_t* states)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i == 0)
        new (world) RenderList(objects, SPHERES_COUNT);
    if(i < SPHERES_COUNT)
    {
        auto position = Vec3(
            curand_normal(states + i) * 20 - 10,
            curand_normal(states + i) * 20 - 10,
            curand_normal(states + i) * 20 - 10
        );

        auto color = Vec3(
            curand_normal(states + i),
            curand_normal(states + i),
            curand_normal(states + i)
        );

        objects[i] = new Sphere(position, curand_normal(states + i), color);
    }
    if(i < LIGHTS_COUNT)
    {
        auto position = Vec3(
            curand_normal(states + i) * 6 - 3,
            curand_normal(states + i) * 6 - 3,
            curand_normal(states + i) * 6 - 3
        );

        new (lights + i) LightSource(position, Vec3::one());
    }
}

__global__ void free_objects(RenderObject** objects)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < SPHERES_COUNT)
        delete objects[i];
}

__global__ void render(RenderList* world, LightSource* lights, Camera camera, int width, int height, float4* pixels)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);
    Ray ray = camera.calculate_ray(u, v);

    Vec3 col = calculate_color(ray, world, lights, camera.position());

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

    const int N = max(SPHERES_COUNT, LIGHTS_COUNT);
    curandState_t* curand_states;
    CUDA_CHECK(cudaMalloc(&curand_states, N * sizeof(curandState_t)));
    init_curand<<<N, 1>>>(time(0), curand_states);
    CUDA_CHECK(cudaDeviceSynchronize());

    RenderList* world;
    RenderObject** objects;
    LightSource* lights;
    CUDA_CHECK(cudaMalloc(&world, sizeof(RenderList)));
    CUDA_CHECK(cudaMalloc(&objects, SPHERES_COUNT * sizeof(RenderObject*)));
    CUDA_CHECK(cudaMalloc(&lights, LIGHTS_COUNT * sizeof(LightSource)));
    dim3 block_size2(32, 1);
    dim3 grid_size2 = calculate_grid_size(N, 1, block_size2);
    create_objects<<<grid_size2, block_size2>>>(objects, lights, world, curand_states);
    CUDA_CHECK(cudaDeviceSynchronize());

    Camera camera(3, Vec3(0, 0, -1), 90, (float)resolution_width / resolution_height);
    float mouse_x = 0, mouse_y = 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
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
        render<<<grid_size, block_size, 0, stream>>>(world, lights, camera, resolution_width, resolution_height, framebuffer);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        renderer.blit();
        renderer.draw();
        display.show();
    }

    free_objects<<<grid_size2, block_size2>>>(objects);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(world));
    CUDA_CHECK(cudaFree(objects));
    CUDA_CHECK(cudaFree(lights));

    return 0;
}
