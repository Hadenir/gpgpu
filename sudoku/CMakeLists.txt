cmake_minimum_required(VERSION 3.20)
project(sudoku CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(.)

add_subdirectory(cpu)

add_executable(sudoku_gpu main.cu board.cuh board.cu solver.cuh solver.cu utils.cuh utils.cu)
set_target_properties(sudoku_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

