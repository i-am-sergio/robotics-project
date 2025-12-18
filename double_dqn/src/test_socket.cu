#include "cuda_utils.h"
#include "dqn_agent.h"
#include "environment.h"
#include "socket_utils.h"  // Nuevo header
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

// Implementaciones de funciones utilitarias
double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int argmax(const std::vector<double>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// --- MAIN PARA PROBAR MODELO CON WEBSOCKET ---
int main(int argc, char* argv[]) {
    // Mostrar info de GPU
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "=== PRUEBA DE MODELO DOUBLE DQN CON WEBSOCKET ===\n";
    std::cout << "GPUs disponibles: " << deviceCount << std::endl;
    
    // Nombre del modelo a cargar (por defecto o desde argumento)
    std::string model_path = "double_dqn_model";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    // Configurar WebSocket
    std::string ws_address = "127.0.0.1";
    int ws_port = 5555;
    
    if (argc > 2) {
        ws_address = argv[2];
    }
    if (argc > 3) {
        ws_port = std::stoi(argv[3]);
    }
    
    std::cout << "Cargando modelo desde: " << model_path << std::endl;
    std::cout << "WebSocket server: ws://" << ws_address << ":" << ws_port << std::endl;
    
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
        
        // Conectar a WebSocket
        WebSocketClient ws_client(ws_address, ws_port);
        bool ws_connected = false;
        
        std::cout << "\nIntentando conectar al servidor WebSocket..." << std::endl;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (ws_client.connect_to_server()) {
                ws_connected = true;
                break;
            }
            std::cout << "Intento " << (attempt + 1) << " fallido, reintentando..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (!ws_connected) {
            std::cout << "⚠️  No se pudo conectar al WebSocket, continuando sin él..." << std::endl;
        }
        
        // Ejecutar múltiples episodios de prueba
        int num_tests = 1;
        int successful_runs = 0;
        
        for (int test = 0; test < num_tests; ++test) {
            std::cout << "\n=== Prueba " << (test + 1) << "/" << num_tests << " ===" << std::endl;
            
            auto state = env.reset();
            bool done = false;
            int steps = 0;
            double total_reward = 0;
            
            std::cout << "Estado inicial:";
            env.render();
            
            // Enviar estado inicial si hay conexión WebSocket
            if (ws_connected) {
                ws_client.send_action("INICIO", 0, 0.0);
            }
            
            while (!done && steps < 30) {
                int action = agent.act(state);
                std::string action_names[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
                std::string action_name = action_names[action];
                
                auto res = env.step(action);
                auto next_state = env.get_encoded_state();
                
                // Mostrar en consola
                std::cout << "Paso " << (steps + 1) << ": " << action_name;
                std::cout << " -> Reward: " << res.first;
                std::cout << " | Posición: (" << env.get_encoded_state()[0] << ")" << std::endl;
                
                // Enviar al servidor WebSocket
                if (ws_connected) {
                    if (!ws_client.send_action(action_name, steps + 1, res.first)) {
                        std::cout << "⚠️  Conexión WebSocket perdida, continuando sin ella..." << std::endl;
                        ws_connected = false;
                    }
                }
                
                state = next_state;
                total_reward += res.first;
                done = res.second;
                steps++;
                
                if (done) {
                    std::string result_msg = (res.first > 0) ? "META" : "TRAMPA";
                    if (ws_connected) {
                        ws_client.send_action(result_msg, steps, total_reward);
                    }
                    
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
                
                // Pequeña pausa para visualización
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (!done) {
                std::cout << "\nLímite de pasos alcanzado. Recompensa total: " << total_reward << std::endl;
                env.render();
                if (ws_connected) {
                    ws_client.send_action("TIMEOUT", steps, total_reward);
                }
            }
            
            std::cout << "Pasos realizados: " << steps << std::endl;
            
            // Pausa entre pruebas
            if (test < num_tests - 1) {
                std::cout << "\nPausa entre pruebas..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        
        // Estadísticas
        std::cout << "\n=== ESTADÍSTICAS FINALES ===" << std::endl;
        std::cout << "Éxitos: " << successful_runs << "/" << num_tests << std::endl;
        std::cout << "Tasa de éxito: " << (100.0 * successful_runs / num_tests) << "%" << std::endl;
        
        // Enviar estadísticas finales
        if (ws_connected) {
            std::string stats_msg = "STATS: " + std::to_string(successful_runs) + "/" + 
                                    std::to_string(num_tests) + " (" + 
                                    std::to_string(100.0 * successful_runs / num_tests) + "%)";
            ws_client.send_action(stats_msg, -1, successful_runs);
        }
        
        // Prueba interactiva adicional
        std::cout << "\n¿Desea realizar una prueba interactiva? (s/n): ";
        char choice;
        std::cin >> choice;
        
        if (choice == 's' || choice == 'S') {
            std::cout << "\n=== PRUEBA INTERACTIVA ===" << std::endl;
            
            // Re-conectar si es necesario
            if (!ws_connected) {
                if (ws_client.connect_to_server()) {
                    ws_connected = true;
                    std::cout << "✅ Reconectado al WebSocket" << std::endl;
                }
            }
            
            auto state = env.reset();
            bool done = false;
            int steps = 0;
            
            std::cout << "Estado inicial:" << std::endl;
            env.render();
            
            if (ws_connected) {
                ws_client.send_action("INTERACTIVO_INICIO", 0, 0.0);
            }
            
            while (!done && steps < 30) {
                std::cout << "\nPresione Enter para siguiente acción...";
                std::cin.ignore();
                std::cin.get();
                
                int action = agent.act(state);
                std::string action_names[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
                std::string action_name = action_names[action];
                
                auto res = env.step(action);
                auto next_state = env.get_encoded_state();
                
                std::cout << "Acción: " << action_name;
                std::cout << " | Reward: " << res.first << std::endl;
                
                // Enviar al WebSocket
                if (ws_connected) {
                    ws_client.send_action(action_name, steps + 1, res.first);
                }
                
                state = next_state;
                done = res.second;
                steps++;
                
                env.render();
                
                if (done) {
                    std::string result_msg = (res.first > 0) ? "META_ALCANZADA" : "TRAMPA_CAIDA";
                    if (ws_connected) {
                        ws_client.send_action(result_msg, steps, res.first);
                    }
                    
                    if (res.first > 0) {
                        std::cout << "\n¡¡¡ META ALCANZADA !!!" << std::endl;
                    } else {
                        std::cout << "\n¡¡¡ CAISTE EN UNA TRAMPA !!!" << std::endl;
                    }
                }
            }
            
            if (!done) {
                std::cout << "\nLímite de pasos alcanzado." << std::endl;
                if (ws_connected) {
                    ws_client.send_action("INTERACTIVO_TIMEOUT", steps, 0.0);
                }
            }
        }
        
        // Desconectar WebSocket
        if (ws_connected) {
            ws_client.disconnect();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== PRUEBA COMPLETADA ===" << std::endl;
    return 0;
}