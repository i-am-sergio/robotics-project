#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include "neural_net.h"
#include <deque>
#include <vector>
#include <utility>

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

#endif // DQN_AGENT_H