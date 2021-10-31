#pragma once

#include <string>
#include <GLFW/glfw3.h>

namespace gfx
{
    class Display
    {
    public:
        Display(const std::string& window_title, int width, int height);

        ~Display();

        size_t get_width() const { return _width; }
        size_t get_height() const { return _height; }

        bool should_close() { return glfwWindowShouldClose(_window); }

        void show();

    private:
        int _width, _height;

        GLFWwindow* _window;

        static Display& get_display_for_window(GLFWwindow* window) { return *(Display*)glfwGetWindowUserPointer(window); }

        // GLFW callbacks
        static void on_error(int error_code, const char* error_message);
        static void on_size_changed(GLFWwindow* window, int new_width, int new_height);
    };
}
