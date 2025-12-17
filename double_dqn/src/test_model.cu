#include "cuda_utils.h"
#include "dqn_agent.h"
#include "environment.h"
#include <iostream>
#include <vector>
#include <string>

// Implementaciones de funciones utilitarias
double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int argmax(const std::vector<double>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// --- MAIN PARA PROBAR MODELO CARGADO ---
int main(int argc, char* argv[]) {
    // Mostrar info de GPU
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "=== PRUEBA DE MODELO DOUBLE DQN ===\n";
    std::cout << "GPUs disponibles: " << deviceCount << std::endl;
    
    // Nombre del modelo a cargar (por defecto o desde argumento)
    std::string model_path = "double_dqn_model";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Cargando modelo desde: " << model_path << std::endl;
    
    try {
        // Cargar agente entrenado
        CudaDoubleDQNAgent agent(model_path);
        
        // Configurar entorno
        int N = 10;
        GridEnvironment env(N);
        
        // Verificar compatibilidad
        if (env.get_state_size() != agent.policy_net.input_size) {
            std::cerr << "Error: Dimensiones incompatibles" << std::endl;
            std::cerr << "Entorno: " << env.get_state_size() << " entradas" << std::endl;
            std::cerr << "Modelo: " << agent.policy_net.input_size << " entradas" << std::endl;
            return 1;
        }
        
        // Configurar para solo explotación (sin exploración)
        agent.epsilon = 0.0;
        
        // Ejecutar múltiples episodios de prueba
        int num_tests = 5;
        int successful_runs = 0;
        
        for (int test = 0; test < num_tests; ++test) {
            std::cout << "\n=== Prueba " << (test + 1) << "/" << num_tests << " ===" << std::endl;
            
            auto state = env.reset();
            bool done = false;
            int steps = 0;
            double total_reward = 0;
            
            std::cout << "Estado inicial:";
            env.render();
            
            while (!done && steps < 30) {
                int action = agent.act(state);
                std::string action_names[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
                
                auto res = env.step(action);
                auto next_state = env.get_encoded_state();
                
                std::cout << "Paso " << (steps + 1) << ": " << action_names[action];
                std::cout << " -> Reward: " << res.first;
                std::cout << " | Posición: (" << env.get_encoded_state()[0] << ")" << std::endl;
                
                state = next_state;
                total_reward += res.first;
                done = res.second;
                steps++;
                
                if (done) {
                    if (res.first > 0) {
                        std::cout << "\n¡META ALCANZADA! Recompensa total: " << total_reward << std::endl;
                        successful_runs++;
                    } else {
                        std::cout << "\n¡CAÍDA EN TRAMPA! Recompensa total: " << total_reward << std::endl;
                    }
                    break;
                }
                
                // Mostrar cada 5 pasos
                if (steps % 5 == 0) {
                    env.render();
                }
            }
            
            if (!done) {
                std::cout << "\nLímite de pasos alcanzado. Recompensa total: " << total_reward << std::endl;
                env.render();
            }
            
            std::cout << "Pasos realizados: " << steps << std::endl;
        }
        
        // Estadísticas
        std::cout << "\n=== ESTADÍSTICAS FINALES ===" << std::endl;
        std::cout << "Éxitos: " << successful_runs << "/" << num_tests << std::endl;
        std::cout << "Tasa de éxito: " << (100.0 * successful_runs / num_tests) << "%" << std::endl;
        
        // Prueba interactiva adicional
        std::cout << "\n¿Desea realizar una prueba interactiva? (s/n): ";
        char choice;
        std::cin >> choice;
        
        if (choice == 's' || choice == 'S') {
            std::cout << "\n=== PRUEBA INTERACTIVA ===" << std::endl;
            auto state = env.reset();
            bool done = false;
            int steps = 0;
            
            env.render();
            
            while (!done && steps < 30) {
                std::cout << "\nPresione Enter para siguiente acción...";
                std::cin.ignore();
                std::cin.get();
                
                int action = agent.act(state);
                std::string action_names[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
                
                auto res = env.step(action);
                auto next_state = env.get_encoded_state();
                
                std::cout << "Acción: " << action_names[action];
                std::cout << " | Reward: " << res.first << std::endl;
                
                state = next_state;
                done = res.second;
                steps++;
                
                env.render();
                
                if (done) {
                    if (res.first > 0) {
                        std::cout << "\n¡¡¡ META ALCANZADA !!!" << std::endl;
                    } else {
                        std::cout << "\n¡¡¡ CAISTE EN UNA TRAMPA !!!" << std::endl;
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}