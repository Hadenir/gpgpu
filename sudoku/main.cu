#include <iostream>
#include <fstream>
#include <iomanip>

#include "solver.cuh"
#include "board.cuh"
#include "utils.cuh"

int* read_file(const std::string& path)
{
    std::ifstream file(path);
    if(!file.is_open())
        throw std::runtime_error("Failed to open file " + path);

    int* cells = new int[9 * 9];

    int i = 0;
    while(!file.eof() && i < 9 * 9)
    {
        int digit;
        file >> digit;
        cells[i++] = digit;
    }

    if(i != 9 * 9)
        throw std::runtime_error("File doesn't contain the whole board");

    return cells;
}

template<typename T>
T* device_malloc(size_t n = 1)
{
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    return ptr;
}

int main(int argc, char* argv[])
{
    if(argc != 2) return 1;

    std::cout << std::setprecision(6);

    std::cout << "Reading board..." << std::endl;
    int* cells = read_file(argv[1]);
    std::cout << "Board loaded:" << std::endl;

    Board board(cells);
    board.print();

    std::cout << "Preparing first solution stage..." << std::endl;

    const int num_bfs_steps = 23;
    const auto max_num_boards = 2 << num_bfs_steps;

    auto* input_boards = device_malloc<Board>(max_num_boards);
    CUDA_CHECK(cudaMemcpy(input_boards, &board, sizeof(Board), cudaMemcpyHostToDevice));
    auto* output_boards = device_malloc<Board>(max_num_boards);
    int* num_output_boards = device_malloc<int>();
    char* empty_indices = device_malloc<char>(9 * 9 * max_num_boards);
    char* num_empty_indices = device_malloc<char>(max_num_boards);

    std::cout << "Performing first stage for " << num_bfs_steps
              << " steps (for max. " << max_num_boards << " boards)..." << std::endl;

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));
    int num_boards = 1;
    for(int i = 0; i < num_bfs_steps; i++)
    {
        if(i % 2 == 0)
            run_sudoku_bfs(input_boards, output_boards, num_boards, num_output_boards, empty_indices, num_empty_indices);
        else
            run_sudoku_bfs(output_boards, input_boards, num_boards, num_output_boards, empty_indices, num_empty_indices);

        CUDA_CHECK(cudaMemcpy(&num_boards, num_output_boards, sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float stage1_time;
    CUDA_CHECK(cudaEventElapsedTime(&stage1_time, start, end));
    std::cout << "First stage completed in " << stage1_time << " ms, generated " << num_boards << " boards."
              << std::endl;

    std::cout << "Preparing second solution stage..." << std::endl;
    if(num_bfs_steps % 2 == 0)
    {
        CUDA_CHECK(cudaFree(output_boards));
        output_boards = input_boards;
    }
    else
    {
        CUDA_CHECK(cudaFree(input_boards));
    }

    Board* boards = output_boards;
    int* complete = device_malloc<int>();
    auto* result = device_malloc<Board>();

    std::cout << "Performing second solution stage for total of " << num_boards << " boards..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    run_sudoku_backtrack(boards, num_boards, empty_indices, num_empty_indices, complete, result);
    CUDA_CHECK(cudaEventRecord(end));

    int cmp;
    CUDA_CHECK(cudaMemcpy(&cmp, complete, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&board, result, sizeof(Board), cudaMemcpyDeviceToHost));

    float stage2_time;
    cudaEventElapsedTime(&stage2_time, start, end);
    std::cout << "Second stage completed in " << stage2_time << " ms." << std::endl;

    if(cmp == 1)
    {
        std::cout << "Correct solution found! Here it is:" << std::endl;
        board.print();
    }
    else
    {
        std::cout << "No correct solution was found :(" << std::endl;
    }

    CUDA_CHECK(cudaFree(complete));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(num_empty_indices));
    CUDA_CHECK(cudaFree(empty_indices));
    CUDA_CHECK(cudaFree(num_output_boards));
    CUDA_CHECK(cudaFree(output_boards));

    delete[] cells;
    return 0;
}