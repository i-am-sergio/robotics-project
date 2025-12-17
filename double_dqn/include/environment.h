#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>
#include <utility>
#include <iostream>

// --- ENTORNO 2D ---
class GridEnvironment {
    int size;
    int px, py, gx, gy;
    std::vector<std::pair<int, int>> traps;

public:
    GridEnvironment(int n) : size(n) {
        gx = n-1; gy = n-1;
        traps = {{1, 1}, {1, 2}, {3, 2}, {3, 3}, {2, 4}}; 
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

#endif // ENVIRONMENT_H