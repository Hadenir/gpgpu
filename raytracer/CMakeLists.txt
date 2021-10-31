cmake_minimum_required(VERSION 3.0.0)
project(raytracer VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;")
find_package(CUDA 11 REQUIRED)
find_package(OpenGL REQUIRED)

add_subdirectory(thirdparty/glad)
add_subdirectory(thirdparty/glfw3)

file(GLOB_RECURSE SOURCE_FILES src/*.cu)
file(GLOB_RECURSE HEADER_FILES src/*.cuh)

add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer glad OpenGL::GL glfw)
set_target_properties(raytracer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/
)
