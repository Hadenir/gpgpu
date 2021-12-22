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

const int LIGHTS_COUNT = 3;
const int SPHERES_COUNT = 50;

Vec3 calculate_color(const Ray& ray, RenderList* world, LightSource* lights, const Vec3& camera_position)
{
    float k_s = 0.4f;
    float k_d = 0.9f;
    float k_a = 0.1f;
    float alpha = 100.0f;
    Vec3 ambient(1, 1, 1);

    HitResult result;
    if(world->hit(ray, 0.0f, FLT_MAX, result))
    {
        auto light_color = k_a * ambient;
        for(int i = 0; i < LIGHTS_COUNT; i++)
        {
            LightSource& light = lights[i];
            auto N = result.normal;
            auto L_m = (light.position() - result.hit_point).normalized(); // direction from surface to light
            auto R_m = 2.0f * L_m.dot(N) * N - L_m; // direction of perfectly reflected ray
            auto V = (camera_position - result.hit_point).normalized(); // direction from surface to the camera

            auto diffuse_intensity = L_m.dot(N);
            if(diffuse_intensity > 0)
                light_color += k_d * diffuse_intensity * light.color();
            auto specular_intensity = R_m.dot(V);
            if(specular_intensity > 0)
                light_color += k_s * powf(specular_intensity, alpha) * light.color();
        }

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

void create_objects(RenderObject** objects, LightSource* lights, RenderList* world)
{
    for(int i = 0; i < SPHERES_COUNT; i++)
    {
        if(i == 0)
            new (world) RenderList(objects, SPHERES_COUNT);
        if(i < SPHERES_COUNT)
        {
            auto position = Vec3(
                (float)rand() / RAND_MAX * 20 - 10,
                (float)rand() / RAND_MAX * 20 - 10,
                (float)rand() / RAND_MAX * 20 - 10
            );

            auto color = Vec3(
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX
            );

            objects[i] = new Sphere(position, (float)rand() / RAND_MAX, color);
        }
        if(i < LIGHTS_COUNT)
        {
            auto position = Vec3(
                (float)rand() / RAND_MAX * 6 - 3,
                (float)rand() / RAND_MAX * 6 - 3,
                (float)rand() / RAND_MAX * 6 - 3
            );

            new (lights + i) LightSource(position, Vec3::one());
        }
    }
}

void render(RenderList* world, LightSource* lights, Camera camera, int width, int height, float4* pixels)
{
    for(uint y = 0; y < height; y++)
    {
        for(uint x = 0; x < width; x++)
        {
            float u = float(x) / float(width - 1);
            float v = float(y) / float(height - 1);
            Ray ray = camera.calculate_ray(u, v);

            Vec3 col = calculate_color(ray, world, lights, camera.position());

            pixels[x + y * width].x = col.r();
            pixels[x + y * width].y = col.g();
            pixels[x + y * width].z = col.b();
        }
    }
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

    // const int N = max(SPHERES_COUNT, LIGHTS_COUNT);
    // curandState_t* curand_states;
    // CUDA_CHECK(cudaMalloc(&curand_states, N * sizeof(curandState_t)));
    // init_curand<<<N, 1>>>(time(0), curand_states);
    // CUDA_CHECK(cudaDeviceSynchronize());

    RenderList* world;
    RenderObject** objects;
    LightSource* lights;
    // CUDA_CHECK(cudaMalloc(&world, sizeof(RenderList)));
    // CUDA_CHECK(cudaMalloc(&objects, SPHERES_COUNT * sizeof(RenderObject*)));
    // CUDA_CHECK(cudaMalloc(&lights, LIGHTS_COUNT * sizeof(LightSource)));
    // dim3 block_size2(32, 1);
    // dim3 grid_size2 = calculate_grid_size(N, 1, block_size2);
    // create_objects<<<grid_size2, block_size2>>>(objects, lights, world, curand_states);
    // CUDA_CHECK(cudaDeviceSynchronize());
    world = (RenderList*)malloc(sizeof(RenderList));
    objects = (RenderObject**)malloc(SPHERES_COUNT * sizeof(RenderObject*));
    lights = (LightSource*)malloc(LIGHTS_COUNT * sizeof(LightSource));
    create_objects(objects, lights, world);

    Camera camera(3, Vec3(0, 0, -1), 90, (float)resolution_width / resolution_height);
    float mouse_x = 0, mouse_y = 0;
    clock_t elapsedTime = 0;
    float4* cpu_framebuffer = new float4[window_width * window_height];
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

        clock_t start = clock() / (CLOCKS_PER_SEC / 1000);
        render(world, lights, camera, resolution_width, resolution_height, cpu_framebuffer);
        
        float4* framebuffer = renderer.get_framebuffer();
        CUDA_CHECK(cudaMemcpy(framebuffer, cpu_framebuffer, window_width * window_height * sizeof(float4), cudaMemcpyHostToDevice));

        clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
        elapsedTime = end - start;
        std::cout << elapsedTime << "ms" << std::endl;

        renderer.blit();
        renderer.draw();
        display.show();
    }

    free(world);
    free(objects);
    free(lights);
    delete[] cpu_framebuffer;

    return 0;
}
