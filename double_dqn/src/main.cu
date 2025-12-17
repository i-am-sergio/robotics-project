#include "cuda_utils.h"
#include "dqn_agent.h"
#include "environment.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm> 
#include <filesystem>

// Implementaciones de funciones utilitarias
double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int argmax(const std::vector<double>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// --- MAIN DE ENTRENAMIENTO ---
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
    CudaDoubleDQNAgent agent(env.get_state_size(), 32, env.get_action_size());

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

    // GUARDAR MODELO ENTRENADO
    std::string model_name = "double_dqn_model";
    if (agent.save(model_name)) {
        std::cout << "\nModelo guardado exitosamente como: " << model_name << "_*" << std::endl;
    } else {
        std::cerr << "\nError al guardar el modelo" << std::endl;
    }

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