#pragma once

class Board
{
public:
    char cells[9 * 9] {};

    __host__ __device__ Board();

    __host__ __device__ explicit Board(int* cells);

    __host__ __device__ bool check() const;

    __host__ __device__ bool check(int index) const;

    __host__ __device__ int find_empty() const;

    __host__ void print() const;
};