#include <stdexcept>
#include <sstream>
#include <glad.h>

#include "display.cuh"

namespace gfx
{
    Display::Display(const std::string& window_title, int width, int height)
        : _width(width), _height(height)
    {
        int result = glfwInit();
        if(result != GLFW_TRUE)
            throw std::runtime_error("Failed to initialize GLFW!");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // TODO: Make window resizeable for extra points.

        _window = glfwCreateWindow(width, height, window_title.c_str(), nullptr, nullptr);
        if(_window == nullptr)
            throw std::runtime_error("Failed to create GLFW window!");

        glfwSetWindowUserPointer(_window, this);
        glfwSetErrorCallback(on_error);
        glfwSetFramebufferSizeCallback(_window, on_size_changed);

        glfwMakeContextCurrent(_window);

        result = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        if(result == 0)
            throw std::runtime_error("Failed to load GLAD!");

        glViewport(0, 0, width, height);
    }

    Display::~Display()
    {
        glfwDestroyWindow(_window);
        glfwTerminate();
    }

    void Display::show()
    {
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }

    void Display::on_error(int error_code, const char* error_message)
    {
        std::stringstream ss;
        ss << "GLFW error (" << error_code << ") occured: " << error_message;
        throw std::runtime_error(ss.str());
    }

    void Display::on_size_changed(GLFWwindow* window, int new_width, int new_height)
    {
        auto display = get_display_for_window(window);
        display._width = new_width;
        display._height = new_height;

        glViewport(0, 0, new_width, new_height);
    }
}
