#include <iostream>

#include "board.cuh"

__host__ __device__ Board::Board()
{}

__host__ __device__ Board::Board(int* cells)
{
    for(int i = 0; i < 9 * 9; i++) this->cells[i] = cells[i];
}

__host__ __device__ bool Board::check() const
{
    bool occurrences[9] {};

    // Check rows.
    for(int r = 0; r < 9; r++)
    {
        memset(occurrences, false, sizeof(occurrences));
        for(int c = 0; c < 9; c++)
        {
            char digit = cells[c + 9 * r];
            if(digit == 0) continue;
            if(occurrences[digit - 1])
                return false;
            occurrences[digit - 1] = true;
        }
    }

    // Check columns.
    for(int c = 0; c < 9; c++)
    {
        memset(occurrences, false, sizeof(occurrences));
        for(int r = 0; r < 9; r++)
        {
            char digit = cells[c + 9 * r];
            if(digit == 0) continue;
            if(occurrences[digit - 1])
                return false;
            occurrences[digit - 1] = true;
        }
    }

    // Check boxes.
    for(int b = 0; b < 9; b++)
    {
        memset(occurrences, false, sizeof(occurrences));
        for(int i = 0; i < 9; i++)
        {
            int c = (b % 3) * 3 + i % 3;
            int r = (b / 3) * 3 + i / 3;
            char digit = cells[c + 9 * r];
            if(digit == 0) continue;
            if(occurrences[digit - 1])
                return false;
            occurrences[digit - 1] = true;
        }
    }

    return true;
}

__host__ __device__ bool Board::check(int index) const
{
    int c = index % 9;
    int r = index / 9;

    bool occurrences[9] {};

    memset(occurrences, false, sizeof(occurrences));
    for(int i = 0; i < 9; i++)
    {
        char digit = cells[i + 9 * r];
        if(digit == 0) continue;
        if(occurrences[digit - 1])
            return false;
        occurrences[digit - 1] = true;
    }

    memset(occurrences, false, sizeof(occurrences));
    for(int i = 0; i < 9; i++)
    {
        char digit = cells[c + 9 * i];
        if(digit == 0) continue;
        if(occurrences[digit - 1])
            return false;
        occurrences[digit - 1] = true;
    }

    memset(occurrences, false, sizeof(occurrences));
    for(int i = 0; i < 9; i++)
    {
        int bc = c / 3;
        int br = r / 3;
        int b = bc + br * 3;
        int x = (b % 3) * 3 + i % 3;
        int y = (b / 3) * 3 + i / 3;
        char digit = cells[x + 9 * y];
        if(digit == 0) continue;
        if(occurrences[digit - 1])
            return false;
        occurrences[digit - 1] = true;
    }

    return true;
}

__host__ void Board::print() const
{
    std::cout << "+-----+-----+-----+\n";
    for(int r = 0; r < 9; r++)
    {
        std::cout << '|';
        for(int c = 0; c < 9; c++)
        {
            std::cout << +cells[c + 9 * r];
            std::cout << (c % 3 == 2 ? '|' : ' ');
        }

        std::cout << (r % 3 == 2 ? "\n+-----+-----+-----+\n" : "\n");
    }
    std::cout << std::endl;
}

__host__ __device__ int Board::find_empty() const
{
    int index = 0;
    while(cells[index] != 0 && index < 9 * 9) index++;
    return index < 9 * 9 ? index : -1;
}

