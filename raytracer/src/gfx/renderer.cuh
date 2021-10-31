#pragma once

namespace gfx
{
    typedef unsigned int uint;

    class Renderer
    {
    public:
        Renderer(int width, int height);

        ~Renderer();

        void clear();

        void blit(const float4* pixels);

        void draw();

    private:
        int _width, _height;

        cudaGraphicsResource_t _cuda_resource;

        uint _shader_program;

        void init_texture();
        void init_shaders();
        void init_vertex_objects();

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
