#include "solver.cuh"
#include "utils.cuh"

static const unsigned int THREADS_PER_BLOCK = 32;
static const unsigned int MAX_NUM_BLOCKS = 2048;

// Kernel przyjmuje liste (częściowo wypelnionych) plansz sudoku.
// Przeszukując wszerz, algorytm generuje listę wszystkich możliwych (poprawnych) plansz, które można uzyskać
// wypełniając pierwszą pustą komórkę we wszystkich planszach wejsciowych.
// Dodatkowo kernel przygotowuje liste indeksów pustych komórek dla każdej wygenerowanej planszy.
__global__ void kernel_sudoku_bfs(
    Board* input_boards,
    Board* output_boards,
    int num_input_boards,
    int* num_output_boards,
    char* empty_indices,
    char* num_empty_indices)
{
    for(unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < num_input_boards;
        idx += blockDim.x * gridDim.x)
    {
        Board& board = input_boards[idx];
        int cell_index = board.find_empty();
        if(cell_index < 0) continue;    // Plansza jest całkowicie wypełniona.

        for(char digit = 1; digit <= 9; digit++)
        {
            board.cells[cell_index] = digit;
            if(board.check(cell_index))     // Wpisanie obecnej cyfry nie lamie zasad sudoku.
            {
                int output_index = atomicAdd(num_output_boards, 1);
                char num_empty = 0;
                for(char i = 0; i < 9 * 9; i++)
                {
                    if(board.cells[i] == 0)
                        empty_indices[num_empty++ + 9 * 9 * output_index] = i;
                }

                num_empty_indices[output_index] = num_empty;
                output_boards[output_index] = board;
            }
        }
    }
}

void run_sudoku_bfs(
    Board* input_boards,
    Board* output_boards,
    int num_input_boards,
    int* num_output_boards,
    char* empty_indices,
    char* num_empty_indices)
{
    CUDA_CHECK(cudaMemset(num_output_boards, 0, sizeof(int)));

    unsigned int num_blocks = num_input_boards / THREADS_PER_BLOCK + 1;
    if(num_blocks > MAX_NUM_BLOCKS) num_blocks = MAX_NUM_BLOCKS;

    kernel_sudoku_bfs<<<num_blocks, THREADS_PER_BLOCK>>>(
        input_boards, output_boards, num_input_boards, num_output_boards, empty_indices, num_empty_indices);
}

// Kernel przjmuje liste czesciowo wypelnionych plansz sudoku, oraz listy indeksów pustych komórek w każdej z tych plansz.
// Za pomocą metody backtrackingu przeszukiwane są wszystkie możliwości wypełnienia planszy zgodnie z zasadami sudoku.
// Jeżeli całkowicie poprawne wypełnienie zostanie znalezione, jest zapisywane jako rezultat.
__global__ void kernel_sudoku_backtrack(
    Board* boards,
    int num_boards,
    char* empty_indices,
    char* num_empty,
    int* complete,
    Board* result)
{
    for(unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        *complete != 1 && idx < num_boards;
        idx += blockDim.x * gridDim.x)
    {
        Board& current_board = boards[idx];
        char* current_empty_indices = empty_indices + idx * 9 * 9;
        char current_num_empty = num_empty[idx];

        int i = 0;
        while(i >= 0 && i < current_num_empty)
        {
            char current_empty = current_empty_indices[i];

            if(current_board.cells[current_empty] >= 9)
            {
                current_board.cells[current_empty] = 0;
                i--;
                continue;
            }

            current_board.cells[current_empty] += 1;
            if(current_board.check(current_empty))
            {
                i++;
            }
        }

        if(i == current_num_empty && atomicAdd(complete, 1) == 0)
        {
            *result = current_board;
        }
    }
}

void run_sudoku_backtrack(
    Board* boards,
    int num_boards,
    char* empty_indices,
    char* num_empty,
    int* complete,
    Board* result)
{
    CUDA_CHECK(cudaMemset(complete, 0, sizeof(int)));

    unsigned int num_blocks = num_boards / THREADS_PER_BLOCK + 1;
    if(num_blocks > MAX_NUM_BLOCKS) num_blocks = MAX_NUM_BLOCKS;

    kernel_sudoku_backtrack<<<num_blocks, THREADS_PER_BLOCK>>>(
        boards, num_boards, empty_indices, num_empty, complete, result);
}
