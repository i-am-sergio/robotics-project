#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Declaraciones de kernels (definidos en kernels.cu)
extern __global__ void argmax_kernel(const double* values, int* result, int size);
extern __global__ void forward_hidden_kernel(const double* input, const double* W1, const double* b1,
                                           double* hidden, int input_size, int hidden_size);
extern __global__ void forward_output_kernel(const double* hidden, const double* W2, const double* b2,
                                           double* output, int hidden_size, int output_size);
extern __global__ void compute_output_delta_kernel(const double* output, const double* target,
                                                 double* output_delta, int output_size);
extern __global__ void compute_hidden_delta_kernel(const double* hidden, const double* output_delta,
                                                 const double* W2, double* hidden_delta,
                                                 int hidden_size, int output_size);
extern __global__ void update_W2_kernel(double* W2, double* b2, const double* hidden,
                                       const double* output_delta, double learning_rate,
                                       int hidden_size, int output_size);
extern __global__ void update_W1_kernel(double* W1, double* b1, const double* input,
                                       const double* hidden_delta, double learning_rate,
                                       int input_size, int hidden_size);
extern __global__ void copy_weights_kernel(const double* src_W1, const double* src_b1,
                                         const double* src_W2, const double* src_b2,
                                         double* dst_W1, double* dst_b1,
                                         double* dst_W2, double* dst_b2,
                                         int input_size, int hidden_size, int output_size);

class CudaNeuralNet {
public:
    int input_size, hidden_size, output_size;
    double learning_rate;
    
    // Host pointers
    double *h_W1, *h_b1, *h_W2, *h_b2;
    double *h_input, *h_hidden, *h_output;
    double *h_output_delta, *h_hidden_delta;
    double *h_target;
    
    // Device pointers
    double *d_W1, *d_b1, *d_W2, *d_b2;
    double *d_input, *d_hidden, *d_output;
    double *d_output_delta, *d_hidden_delta;
    double *d_target;
    
    // Para argmax
    int *d_argmax_result;

    // Métodos para guardar/cargar
    bool save_to_file(const std::string& filename);
    bool load_from_file(const std::string& filename);
    void save_weights_to_host();  // Helper para sincronizar GPU->CPU
    
    // Constructor alternativo para carga
    CudaNeuralNet(const std::string& filename);
    
    CudaNeuralNet(int i_size, int h_size, int o_size, double lr);
    ~CudaNeuralNet();
    
    void copy_weights_from(const CudaNeuralNet& other);
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input);
    int forward_argmax(const std::vector<double>& input);
    double get_q_value(const std::vector<double>& input, int action);
    void train_step(const std::vector<double>& input, const std::vector<double>& target);
};

// Implementaciones inline
inline CudaNeuralNet::CudaNeuralNet(int i_size, int h_size, int o_size, double lr) 
    : input_size(i_size), hidden_size(h_size), output_size(o_size), learning_rate(lr) {
    
    // Allocate host memory
    h_W1 = new double[input_size * hidden_size];
    h_b1 = new double[hidden_size];
    h_W2 = new double[hidden_size * output_size];
    h_b2 = new double[output_size];
    
    h_input = new double[input_size];
    h_hidden = new double[hidden_size];
    h_output = new double[output_size];
    h_output_delta = new double[output_size];
    h_hidden_delta = new double[hidden_size];
    h_target = new double[output_size];
    
    // Inicializacion Xavier
    std::mt19937 rng(std::random_device{}());
    double limit1 = sqrt(6.0 / (input_size + hidden_size));
    double limit2 = sqrt(6.0 / (hidden_size + output_size));
    std::uniform_real_distribution<double> dist1(-limit1, limit1);
    std::uniform_real_distribution<double> dist2(-limit2, limit2);
    
    for(int i = 0; i < input_size * hidden_size; ++i) h_W1[i] = dist1(rng);
    for(int i = 0; i < hidden_size; ++i) h_b1[i] = 0.0;
    for(int i = 0; i < hidden_size * output_size; ++i) h_W2[i] = dist2(rng);
    for(int i = 0; i < output_size; ++i) h_b2[i] = 0.0;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_W1, input_size * hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b1, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_W2, hidden_size * output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b2, output_size * sizeof(double)));
    
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_output_delta, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_hidden_delta, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_target, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_argmax_result, sizeof(int)));
    
    // Copy initial weights to device
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, input_size * hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, hidden_size * output_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, output_size * sizeof(double), cudaMemcpyHostToDevice));
}

inline CudaNeuralNet::~CudaNeuralNet() {
    // Free host memory
    delete[] h_W1; delete[] h_b1; delete[] h_W2; delete[] h_b2;
    delete[] h_input; delete[] h_hidden; delete[] h_output;
    delete[] h_output_delta; delete[] h_hidden_delta; delete[] h_target;
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_W1)); CHECK_CUDA(cudaFree(d_b1));
    CHECK_CUDA(cudaFree(d_W2)); CHECK_CUDA(cudaFree(d_b2));
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_output)); CHECK_CUDA(cudaFree(d_output_delta));
    CHECK_CUDA(cudaFree(d_hidden_delta)); CHECK_CUDA(cudaFree(d_target));
    CHECK_CUDA(cudaFree(d_argmax_result));
}

inline void CudaNeuralNet::copy_weights_from(const CudaNeuralNet& other) {
    int block_size = 256;
    int total_size = std::max({input_size * hidden_size, hidden_size, 
                              hidden_size * output_size, output_size});
    int grid_size = (total_size + block_size - 1) / block_size;
    
    copy_weights_kernel<<<grid_size, block_size>>>(
        other.d_W1, other.d_b1, other.d_W2, other.d_b2,
        d_W1, d_b1, d_W2, d_b2,
        input_size, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Tambien copiar a host para consistencia
    CHECK_CUDA(cudaMemcpy(h_W1, d_W1, input_size * hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1, d_b1, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, d_W2, hidden_size * output_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2, d_b2, output_size * sizeof(double), cudaMemcpyDeviceToHost));
}

inline std::pair<std::vector<double>, std::vector<double>> CudaNeuralNet::forward(const std::vector<double>& input) {
    // Copy input to device
    for(int i = 0; i < input_size; ++i) h_input[i] = input[i];
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Configure kernel launches
    int block_size = 256;
    
    // Forward to hidden layer
    int grid_size_hidden = (hidden_size + block_size - 1) / block_size;
    forward_hidden_kernel<<<grid_size_hidden, block_size>>>(
        d_input, d_W1, d_b1, d_hidden, input_size, hidden_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Forward to output layer
    int grid_size_output = (output_size + block_size - 1) / block_size;
    forward_output_kernel<<<grid_size_output, block_size>>>(
        d_hidden, d_W2, d_b2, d_output, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_hidden, d_hidden, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    std::vector<double> hidden_vec(h_hidden, h_hidden + hidden_size);
    std::vector<double> output_vec(h_output, h_output + output_size);
    
    return {hidden_vec, output_vec};
}

inline int CudaNeuralNet::forward_argmax(const std::vector<double>& input) {
    // Realizar forward pass
    forward(input);
    
    // Ejecutar argmax en GPU
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;
    int* h_argmax_result = new int[grid_size];
    int* d_temp_result;
    CHECK_CUDA(cudaMalloc(&d_temp_result, grid_size * sizeof(int)));
    
    argmax_kernel<<<grid_size, block_size>>>(d_output, d_temp_result, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(h_argmax_result, d_temp_result, grid_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Encontrar el maximo global
    int best_idx = 0;
    double best_value = -INFINITY;
    for (int i = 0; i < grid_size; ++i) {
        int idx = h_argmax_result[i];
        if (idx >= 0 && idx < output_size) {
            if (h_output[idx] > best_value) {
                best_value = h_output[idx];
                best_idx = idx;
            }
        }
    }
    
    delete[] h_argmax_result;
    CHECK_CUDA(cudaFree(d_temp_result));
    
    return best_idx;
}

inline double CudaNeuralNet::get_q_value(const std::vector<double>& input, int action) {
    forward(input);
    return h_output[action];
}

inline void CudaNeuralNet::train_step(const std::vector<double>& input, const std::vector<double>& target) {
    // Copy input and target to device
    for(int i = 0; i < input_size; ++i) h_input[i] = input[i];
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target, target.data(), output_size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Forward pass
    int block_size = 256;
    
    // Hidden layer forward
    int grid_size_hidden = (hidden_size + block_size - 1) / block_size;
    forward_hidden_kernel<<<grid_size_hidden, block_size>>>(
        d_input, d_W1, d_b1, d_hidden, input_size, hidden_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Output layer forward
    int grid_size_output = (output_size + block_size - 1) / block_size;
    forward_output_kernel<<<grid_size_output, block_size>>>(
        d_hidden, d_W2, d_b2, d_output, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Compute output delta
    compute_output_delta_kernel<<<grid_size_output, block_size>>>(
        d_output, d_target, d_output_delta, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Compute hidden delta
    compute_hidden_delta_kernel<<<grid_size_hidden, block_size>>>(
        d_hidden, d_output_delta, d_W2, d_hidden_delta, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Update weights and biases
    int grid_size_W2 = (hidden_size * output_size + block_size - 1) / block_size;
    update_W2_kernel<<<grid_size_W2, block_size>>>(
        d_W2, d_b2, d_hidden, d_output_delta, learning_rate, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());
    
    int grid_size_W1 = (input_size * hidden_size + block_size - 1) / block_size;
    update_W1_kernel<<<grid_size_W1, block_size>>>(
        d_W1, d_b1, d_input, d_hidden_delta, learning_rate, input_size, hidden_size);
    CHECK_CUDA(cudaGetLastError());
    
    // Synchronize and update host weights
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Actualizar pesos en host para consistencia
    CHECK_CUDA(cudaMemcpy(h_W1, d_W1, input_size * hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, d_W2, hidden_size * output_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1, d_b1, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2, d_b2, output_size * sizeof(double), cudaMemcpyDeviceToHost));
}

inline bool CudaNeuralNet::save_to_file(const std::string& filename) {
    // Sincronizar pesos de GPU a CPU
    save_weights_to_host();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << " para escritura" << std::endl;
        return false;
    }
    
    // Guardar metadatos
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    
    // Guardar pesos
    file.write(reinterpret_cast<const char*>(h_W1), input_size * hidden_size * sizeof(double));
    file.write(reinterpret_cast<const char*>(h_b1), hidden_size * sizeof(double));
    file.write(reinterpret_cast<const char*>(h_W2), hidden_size * output_size * sizeof(double));
    file.write(reinterpret_cast<const char*>(h_b2), output_size * sizeof(double));
    
    file.close();
    std::cout << "Modelo guardado en: " << filename << std::endl;
    return true;
}

inline bool CudaNeuralNet::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << " para lectura" << std::endl;
        return false;
    }
    
    // Leer metadatos
    int loaded_input, loaded_hidden, loaded_output;
    double loaded_lr;
    
    file.read(reinterpret_cast<char*>(&loaded_input), sizeof(loaded_input));
    file.read(reinterpret_cast<char*>(&loaded_hidden), sizeof(loaded_hidden));
    file.read(reinterpret_cast<char*>(&loaded_output), sizeof(loaded_output));
    file.read(reinterpret_cast<char*>(&loaded_lr), sizeof(loaded_lr));
    
    // Verificar compatibilidad
    if (loaded_input != input_size || loaded_hidden != hidden_size || loaded_output != output_size) {
        std::cerr << "Error: Dimensiones del modelo no coinciden" << std::endl;
        std::cerr << "Esperado: " << input_size << "x" << hidden_size << "x" << output_size << std::endl;
        std::cerr << "Cargado: " << loaded_input << "x" << loaded_hidden << "x" << loaded_output << std::endl;
        return false;
    }
    
    // Leer pesos
    file.read(reinterpret_cast<char*>(h_W1), input_size * hidden_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_b1), hidden_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_W2), hidden_size * output_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_b2), output_size * sizeof(double));
    
    file.close();
    
    // Copiar pesos al dispositivo
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, input_size * hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, hidden_size * output_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, output_size * sizeof(double), cudaMemcpyHostToDevice));
    
    std::cout << "Modelo cargado desde: " << filename << std::endl;
    return true;
}

inline void CudaNeuralNet::save_weights_to_host() {
    // Copiar pesos de GPU a CPU
    CHECK_CUDA(cudaMemcpy(h_W1, d_W1, input_size * hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1, d_b1, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, d_W2, hidden_size * output_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2, d_b2, output_size * sizeof(double), cudaMemcpyDeviceToHost));
}

// Constructor para cargar desde archivo
inline CudaNeuralNet::CudaNeuralNet(const std::string& filename) 
    : input_size(0), hidden_size(0), output_size(0), learning_rate(0.0) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo del modelo: " + filename);
    }
    
    // Leer metadatos
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    
    // Allocate host memory
    h_W1 = new double[input_size * hidden_size];
    h_b1 = new double[hidden_size];
    h_W2 = new double[hidden_size * output_size];
    h_b2 = new double[output_size];
    
    h_input = new double[input_size];
    h_hidden = new double[hidden_size];
    h_output = new double[output_size];
    h_output_delta = new double[output_size];
    h_hidden_delta = new double[hidden_size];
    h_target = new double[output_size];
    
    // Leer pesos
    file.read(reinterpret_cast<char*>(h_W1), input_size * hidden_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_b1), hidden_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_W2), hidden_size * output_size * sizeof(double));
    file.read(reinterpret_cast<char*>(h_b2), output_size * sizeof(double));
    
    file.close();
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_W1, input_size * hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b1, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_W2, hidden_size * output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b2, output_size * sizeof(double)));
    
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_output_delta, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_hidden_delta, hidden_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_target, output_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_argmax_result, sizeof(int)));
    
    // Copiar pesos al dispositivo
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, input_size * hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, hidden_size * output_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, output_size * sizeof(double), cudaMemcpyHostToDevice));
    
    std::cout << "Modelo cargado desde archivo: " << filename << std::endl;
    std::cout << "Dimensiones: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
}

#endif // NEURAL_NET_H