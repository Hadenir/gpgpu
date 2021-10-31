#pragma once

#include <sstream>

#define CUDA_CHECK(result) cuda_check((result), #result, __FILE__, __LINE__)
void cuda_check(cudaError_t result, const char* func, const char* file, const int line);
