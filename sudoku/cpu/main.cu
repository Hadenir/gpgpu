#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "board.cuh"

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

bool recursive_backtrack(Board& board, int index = 0)
{
    if(board.check())
    {
        while(board.cells[index] != 0 && index < 9 * 9) index++;

        if(index >= 9 * 9) return true;

        for(char digit = 1; digit <= 9; digit++)
        {
            board.cells[index] = digit;

            if(recursive_backtrack(board, index + 1))
                return true;

            board.cells[index] = 0;
        }
    }

    return false;
}

int main(int argc, char* argv[])
{
    if(argc != 2) return 1;

    std::cout << std::setprecision(6);

    int* cells = read_file(argv[1]);

    Board board(cells);
    board.print();

    auto begin = std::chrono::steady_clock::now();
    bool result = recursive_backtrack(board);
    auto end = std::chrono::steady_clock::now();

    long long time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Algorithm completed in " << (float) time / 1000 << " ms." << std::endl;

    if(result)
    {
        std::cout << "Solution found: " << std::endl;
        board.print();
    }
    else
    {
        std::cout << "No correct solution found :(" << std::endl;
    }

    delete[] cells;
    return 0;
}
