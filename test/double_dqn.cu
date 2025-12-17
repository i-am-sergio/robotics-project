#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <deque>
#include <iomanip>
#include <chrono>

// --- UTILIDADES CUDA ---
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error en " << #call << ": " << cudaGetErrorString(err) << \
        " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__host__ __device__ double relu(double x) {
    return (x > 0) ? x : 0.0;
}

__host__ __device__ double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

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

// --- RED NEURONAL CON CUDA ---
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
    
    CudaNeuralNet(int i_size, int h_size, int o_size, double lr) 
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
    
    ~CudaNeuralNet() {
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
    
    void copy_weights_from(const CudaNeuralNet& other) {
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
    
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input) {
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
    
    // Version de forward que devuelve el indice de la accion maxima (GPU accelerated)
    int forward_argmax(const std::vector<double>& input) {
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
    
    // Obtener Q-value especifico
    double get_q_value(const std::vector<double>& input, int action) {
        forward(input);
        return h_output[action];
    }
    
    void train_step(const std::vector<double>& input, const std::vector<double>& target) {
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
};

// --- UTILIDADES CPU ---
double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int argmax(const std::vector<double>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// --- ESTRUCTURAS ---
struct Transition {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

// --- AGENTE DOUBLE DQN CON CUDA ---
class CudaDoubleDQNAgent {
public:
    CudaNeuralNet policy_net; // Red Online
    CudaNeuralNet target_net; // Red Target
    
    double gamma = 0.95;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.9995;
    
    std::deque<Transition> memory;
    int batch_size = 64;
    int max_memory = 5000;
    
    int step_count = 0;
    int target_update_freq = 200; // Actualizar Target cada 200 pasos

    CudaDoubleDQNAgent(int input, int hidden, int output) 
        : policy_net(input, hidden, output, 0.005), 
          target_net(input, hidden, output, 0.0) {
        update_target_network();
    }

    void update_target_network() {
        target_net.copy_weights_from(policy_net);
    }

    int act(const std::vector<double>& state) {
        if (random_double(0, 1) <= epsilon) {
            return (int)random_double(0, policy_net.output_size - 0.01);
        }
        
        // Usar GPU para forward y argmax
        return policy_net.forward_argmax(state);
    }

    void remember(std::vector<double> s, int a, double r, std::vector<double> ns, bool d) {
        if (memory.size() >= max_memory) memory.pop_front();
        memory.push_back({s, a, r, ns, d});
    }

    void replay() {
        if (memory.size() < batch_size) return;
        
        // Muestreo aleatorio del replay buffer
        std::vector<Transition> batch;
        for(int i = 0; i < batch_size; i++) {
            int idx = (int)random_double(0, memory.size() - 1);
            batch.push_back(memory[idx]);
        }

        // Entrenamiento por lotes
        for (const auto& t : batch) {
            // 1. Obtener Q-values actuales de Policy Net
            auto fwd = policy_net.forward(t.state);
            std::vector<double> target_vector = fwd.second; 
            
            double q_update = t.reward;
            
            if (!t.done) {
                // --- LOGICA DOUBLE DQN ---
                
                // A. Seleccion de accion: Policy Net elige mejor accion en S_next
                int best_action_next = policy_net.forward_argmax(t.next_state);
                
                // B. Evaluacion: Target Net evalua Q-value de esa accion
                double q_value_target = target_net.get_q_value(t.next_state, best_action_next);
                
                // C. Ecuacion de Bellman
                q_update += gamma * q_value_target;
            }
            
            // Actualizar solo el target para la accion tomada
            target_vector[t.action] = q_update;
            
            // Entrenar solo la Policy Net
            policy_net.train_step(t.state, target_vector);
        }

        // Decrementar epsilon
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;

        // Actualizacion periodica de Target Network
        step_count++;
        if (step_count % target_update_freq == 0) {
            update_target_network();
        }
    }
};

// --- ENTORNO 2D ---
class GridEnvironment {
    int size;
    int px, py, gx, gy;
    std::vector<std::pair<int, int>> traps;

public:
    GridEnvironment(int n) : size(n) {
        gx = n-1; gy = n-1;
        traps = {{1, 1}, {1, 2}, {3, 3}, {2, 4},
                 {4, 1}, {4, 3}, {2, 3}, {5, 1}}; 
    }

    int get_state_size() { return size * size; }
    int get_action_size() { return 4; }

    std::vector<double> reset() {
        px = 0; py = 0;
        return get_encoded_state();
    }

    std::vector<double> get_encoded_state() {
        std::vector<double> state(size * size, 0.0);
        state[py * size + px] = 1.0;
        return state;
    }

    std::pair<double, bool> step(int action) {
        int old_x = px; int old_y = py;
        if (action == 0) py--;
        else if (action == 1) py++;
        else if (action == 2) px--;
        else if (action == 3) px++;

        if (px < 0) px = 0; 
        if (px >= size) px = size - 1;
        if (py < 0) py = 0; 
        if (py >= size) py = size - 1;

        if (px == old_x && py == old_y) return {-2.0, false}; // Pared
        if (px == gx && py == gy) return {20.0, true};        // Meta
        for(auto t : traps) if (px == t.first && py == t.second) return {-20.0, true}; // Trampa

        return {-1.0, false}; // Paso normal
    }
    
    void render() {
        std::cout << "\n";
        for(int y = 0; y < size; ++y) {
            for(int x = 0; x < size; ++x) {
                if(x == px && y == py) std::cout << "A ";
                else if(x == gx && y == gy) std::cout << "G ";
                else {
                    bool tr = false; 
                    for(auto t : traps) if(t.first == x && t.second == y) tr = true;
                    std::cout << (tr ? "X " : ". ");
                }
            }
            std::cout << "\n";
        }
        std::cout << "Posicion: (" << px << "," << py << ")\n";
    }
};

// --- MAIN ---
int main() {
    // Mostrar info de GPU
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "=== DOUBLE DQN CON CUDA ===\n";
    std::cout << "GPUs disponibles: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Memoria: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    }
    
    int N = 10;
    GridEnvironment env(N);
    CudaDoubleDQNAgent agent(env.get_state_size(), 64, env.get_action_size());

    int episodes = 1000;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n--- Entrenamiento Double DQN (" << N << "x" << N << ") ---\n";
    
    for (int e = 0; e < episodes; ++e) {
        auto state = env.reset();
        double total_reward = 0;
        bool done = false;
        int steps = 0;

        while (!done && steps < 50) {
            int action = agent.act(state);
            auto res = env.step(action);
            auto next_state = env.get_encoded_state();
            
            agent.remember(state, action, res.first, next_state, res.second);
            agent.replay();

            state = next_state;
            total_reward += res.first;
            done = res.second;
            steps++;
        }

        if ((e + 1) % 100 == 0) {
            std::cout << "Ep: " << std::setw(4) << e + 1 
                      << " | Reward: " << std::setw(5) << total_reward 
                      << " | Epsilon: " << std::fixed << std::setprecision(3) << agent.epsilon 
                      << " | Memory: " << agent.memory.size() 
                      << " | TargetUpdates: " << agent.step_count / agent.target_update_freq << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nTiempo de entrenamiento: " << duration.count() / 1000.0 << " segundos\n";

    // DEMOSTRACION FINAL
    std::cout << "\n--- TEST FINAL (Double DQN) ---\n";
    agent.epsilon = 0.0;
    auto state = env.reset();
    bool done = false;
    
    std::cout << "\nEstado inicial:";
    env.render();
    
    int steps = 0;
    while(!done && steps < 20) {
        int action = agent.act(state);
        std::string action_names[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
        
        auto res = env.step(action);
        std::cout << "Paso " << steps + 1 << ": " << action_names[action];
        std::cout << " -> Reward: " << res.first << std::endl;
        
        state = env.get_encoded_state();
        done = res.second;
        env.render();
        steps++;
        
        if(done) {
            if(res.first > 0) std::cout << "\n¡¡¡ META ALCANZADA !!!\n";
            else std::cout << "\n¡¡¡ CAISTE EN UNA TRAMPA !!!\n";
        }
    }
    
    if (!done) {
        std::cout << "\nLimite de pasos alcanzado.\n";
    }

    return 0;
}