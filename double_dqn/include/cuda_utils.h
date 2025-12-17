#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <cuda_runtime.h>

// --- MACRO PARA VERIFICAR ERRORES CUDA ---
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error en " << #call << ": " << cudaGetErrorString(err) << \
        " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// --- FUNCIONES ACTIVACION ---
__host__ __device__ inline double relu(double x) {
    return (x > 0) ? x : 0.0;
}

__host__ __device__ inline double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// --- UTILIDADES CPU ---
double random_double(double min, double max);
int argmax(const std::vector<double>& v);

#endif // CUDA_UTILS_H