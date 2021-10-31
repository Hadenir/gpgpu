#pragma once

namespace gfx
{
    typedef unsigned int uint;

    class Renderer
    {
    public:
        Renderer(int width, int height);

        ~Renderer();

        float4* get_framebuffer();

        void clear();

        void blit();

        void draw();

    private:
        int _width, _height;

        float4* _framebuffer = nullptr;

        cudaGraphicsResource_t _cuda_resource;
        uint _shader_program;

        void init_texture();
        void init_shaders();
        void init_vertex_objects();

        void allocate_framebuffer();

        const float CLEAR_COLOR[4] = {0.2f, 0.3f, 0.3f, 1.0f};
        const float VERTICES_DATA[30] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };

        static const std::string VERTEX_SHADER_SOURCE;
        static const std::string FRAGMENT_SHADER_SOURCE;
    };
}
