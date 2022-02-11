#pragma once

#include "board.cuh"

void run_sudoku_bfs(
    Board* input_boards,
    Board* output_boards,
    int num_input_boards,
    int* num_output_boards,
    char* empty_indices,
    char* num_empty_indices);

void run_sudoku_backtrack(
    Board* boards,
    int num_boards,
    char* empty_indices,
    char* num_empty,
    int* complete,
    Board* result);