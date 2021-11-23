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

        void get_cursor_pos(float& x, float& y) const;

        bool is_dragging() const;

        Display(const Display&) = delete;
        Display& operator=(Display&) = delete;

    private:
        int _width, _height;

        GLFWwindow* _window;

        float _mouse_x, _mouse_y;
        bool _dragging = false;

        static Display& get_display_for_window(GLFWwindow* window) { return *(Display*)glfwGetWindowUserPointer(window); }

        // GLFW callbacks
        static void on_error(int error_code, const char* error_message);
        static void on_size_changed(GLFWwindow* window, int new_width, int new_height);
        static void on_mouse_moved(GLFWwindow* window, double x_pos, double y_pos);
        static void on_mouse_button(GLFWwindow* window, int button, int action, int mods);
    };
}
