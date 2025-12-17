#ifndef KERNELS_CU
#define KERNELS_CU

#include "cuda_utils.h"
#include <cmath>

// Kernel para encontrar el indice del maximo valor
__global__ void argmax_kernel(const double* values, int* result, int size) {
    __shared__ double shared_max[256];
    __shared__ int shared_idx[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    double local_max = -INFINITY;
    int local_idx = -1;
    
    if (i < size) {
        local_max = values[i];
        local_idx = i;
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduccion en paralelo
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = shared_idx[0];
    }
}

// --- KERNELS PARA FORWARD PASS ---
__global__ void forward_hidden_kernel(const double* input, const double* W1, const double* b1,
                                     double* hidden, int input_size, int hidden_size) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < hidden_size) {
        double sum = 0.0;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * W1[i * hidden_size + h];
        }
        hidden[h] = relu(sum + b1[h]);
    }
}

__global__ void forward_output_kernel(const double* hidden, const double* W2, const double* b2,
                                     double* output, int hidden_size, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < output_size) {
        double sum = 0.0;
        for (int h = 0; h < hidden_size; ++h) {
            sum += hidden[h] * W2[h * output_size + o];
        }
        output[o] = sum + b2[o];
    }
}

// --- KERNELS PARA BACKPROPAGATION ---
__global__ void compute_output_delta_kernel(const double* output, const double* target,
                                           double* output_delta, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < output_size) {
        output_delta[o] = output[o] - target[o];
    }
}

__global__ void compute_hidden_delta_kernel(const double* hidden, const double* output_delta,
                                           const double* W2, double* hidden_delta,
                                           int hidden_size, int output_size) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < hidden_size) {
        double error = 0.0;
        for (int o = 0; o < output_size; ++o) {
            error += output_delta[o] * W2[h * output_size + o];
        }
        hidden_delta[h] = error * relu_derivative(hidden[h]);
    }
}

__global__ void update_W2_kernel(double* W2, double* b2, const double* hidden,
                                 const double* output_delta, double learning_rate,
                                 int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = hidden_size * output_size;
    
    if (idx < total_threads) {
        int h = idx / output_size;
        int o = idx % output_size;
        
        W2[h * output_size + o] -= learning_rate * output_delta[o] * hidden[h];
        
        // Solo un thread por output actualiza el bias
        if (h == 0) {
            atomicAdd(&b2[o], -learning_rate * output_delta[o]);
        }
    }
}

__global__ void update_W1_kernel(double* W1, double* b1, const double* input,
                                 const double* hidden_delta, double learning_rate,
                                 int input_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = input_size * hidden_size;
    
    if (idx < total_threads) {
        int i = idx / hidden_size;
        int h = idx % hidden_size;
        
        W1[i * hidden_size + h] -= learning_rate * hidden_delta[h] * input[i];
        
        // Solo un thread por hidden neuron actualiza el bias
        if (i == 0) {
            atomicAdd(&b1[h], -learning_rate * hidden_delta[h]);
        }
    }
}

// Kernel para copiar pesos entre redes
__global__ void copy_weights_kernel(const double* src_W1, const double* src_b1,
                                   const double* src_W2, const double* src_b2,
                                   double* dst_W1, double* dst_b1,
                                   double* dst_W2, double* dst_b2,
                                   int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copiar W1
    int w1_size = input_size * hidden_size;
    if (idx < w1_size) {
        dst_W1[idx] = src_W1[idx];
    }
    
    // Copiar b1
    if (idx < hidden_size) {
        dst_b1[idx] = src_b1[idx];
    }
    
    // Copiar W2
    int w2_size = hidden_size * output_size;
    if (idx < w2_size) {
        dst_W2[idx] = src_W2[idx];
    }
    
    // Copiar b2
    if (idx < output_size) {
        dst_b2[idx] = src_b2[idx];
    }
}

#endif // KERNELS_CU