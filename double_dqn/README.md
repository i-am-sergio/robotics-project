# Double DQN con CUDA y WebSockets

## 1. Introducción al Concepto: Double DQN

El algoritmo implementado es una variante del **Deep Q-Network (DQN)** conocida como **Double DQN**. En el aprendizaje por refuerzo estándar (Q-Learning o DQN simple), el agente tiende a sobreestimar los valores Q debido a que utiliza el mismo estimador (la misma red neuronal) tanto para seleccionar la mejor acción como para evaluar su valor.

Esta implementación mitiga ese sesgo desacoplando la selección y la evaluación mediante dos redes neuronales:

1. **Policy Network (Red de Política):** Se utiliza para seleccionar la acción óptima para el siguiente estado ().
2. **Target Network (Red Objetivo):** Se utiliza para calcular el valor Q de dicha acción ().

La ecuación de actualización de Bellman utilizada en este código es:

El proyecto utiliza **CUDA** para paralelizar los cálculos de propagación hacia adelante (forward pass) y retropropagación (backpropagation), acelerando significativamente el entrenamiento en comparación con una implementación solo en CPU.

---

## 2. Análisis de Archivos y Funciones

### 2.1. Utilidades Base (`cuda_utils.h`)

Este archivo contiene macros para el manejo de errores de CUDA y funciones auxiliares matemáticas.

#### Macro CHECK_CUDA y Funciones Auxiliares

```cpp
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error en " << #call << ": " << cudaGetErrorString(err) << \
        " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__host__ __device__ inline double relu(double x) {
    return (x > 0) ? x : 0.0;
}

__host__ __device__ inline double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

double random_double(double min, double max);
int argmax(const std::vector<double>& v);

#endif // CUDA_UTILS_H
```

- **CHECK_CUDA:** Es una macro fundamental para la depuración. Envuelve cualquier llamada a la API de CUDA. Si la llamada falla, imprime el error específico, el archivo y la línea donde ocurrió, y termina el programa.

- **relu / relu_derivative:** Implementan la función de activación ReLU (Rectified Linear Unit) y su derivada. Están marcadas con `__host__ __device__` para ser ejecutables tanto en CPU como en GPU.

### 2.2. Definición del Entorno (`environment.h`)

Define el mundo donde interactúa el agente. En este caso, un entorno de rejilla (GridWorld).

#### Clase GridEnvironment

```cpp
class GridEnvironment {
    int size;
    int px, py, gx, gy;
    std::vector<std::pair<int, int>> traps;

public:
    GridEnvironment(int n) : size(n) {
        gx = n-1; gy = n-1;
        traps = {{1, 1}, {3, 2}, {3, 3}, {2, 4},
                 {5, 5}, {5, 6}, {7, 8}, {6, 8}};
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

        if (px == old_x && py == old_y) return {-2.0, false};
        if (px == gx && py == gy) return {20.0, true};
        for(auto t : traps) if (px == t.first && py == t.second) return {-20.0, true};
        return {-1.0, false};
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
```

- **Estado:** El entorno es una cuadrícula de . El estado se representa como un vector "one-hot" (todo ceros excepto la posición actual del agente).
- **step(action):** Procesa el movimiento (0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha). Calcula las colisiones con los bordes y determina la recompensa:
- **+20.0:** Llegar a la meta (G).
- **-20.0:** Caer en una trampa (X).
- **-2.0:** Chocar con una pared.
- **-1.0:** Paso normal (penalización por tiempo para incentivar caminos cortos).

### 2.3. Red Neuronal en CUDA (`neural_net.h`)

Gestiona la memoria en GPU y define la interfaz para los kernels de CUDA.

#### Clase CudaNeuralNet (Declaración y Constructor)

```cpp
class CudaNeuralNet {
public:
    int input_size, hidden_size, output_size;
    double learning_rate;

    double *h_W1, *h_b1, *h_W2, *h_b2;
    // ... (otros punteros host)
    double *d_W1, *d_b1, *d_W2, *d_b2;
    // ... (otros punteros device)

    // ... (Constructores y Destructores)

    inline CudaNeuralNet::CudaNeuralNet(int i_size, int h_size, int o_size, double lr)
        : input_size(i_size), hidden_size(h_size), output_size(o_size), learning_rate(lr) {

        // Asignación de memoria en Host
        h_W1 = new double[input_size * hidden_size];
        // ... (inicialización de otros arrays host)

        // Inicialización de pesos (Xavier Initialization)
        std::mt19937 rng(std::random_device{}());
        double limit1 = sqrt(6.0 / (input_size + hidden_size));
        // ... (lógica de inicialización aleatoria)

        // Asignación de memoria en Device (GPU)
        CHECK_CUDA(cudaMalloc(&d_W1, input_size * hidden_size * sizeof(double)));
        // ... (otros mallocs en CUDA)

        // Copia inicial de pesos Host -> Device
        CHECK_CUDA(cudaMemcpy(d_W1, h_W1, input_size * hidden_size * sizeof(double), cudaMemcpyHostToDevice));
    }
};

```

La clase encapsula una red neuronal de una capa oculta.

- Mantiene copias de los pesos tanto en la CPU (Host) como en la GPU (Device).
- El constructor inicializa los pesos utilizando la técnica **Xavier/Glorot**, ideal para mantener la varianza de las activaciones a través de las capas.
- Utiliza `cudaMalloc` y `cudaMemcpy` para preparar el entorno de la GPU antes del entrenamiento.

#### Funciones de Propagación (Forward)

```cpp
inline std::pair<std::vector<double>, std::vector<double>> CudaNeuralNet::forward(const std::vector<double>& input) {
    // Copiar entrada a GPU
    for(int i = 0; i < input_size; ++i) h_input[i] = input[i];
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(double), cudaMemcpyHostToDevice));

    int block_size = 256;

    // Kernel Capa Oculta
    int grid_size_hidden = (hidden_size + block_size - 1) / block_size;
    forward_hidden_kernel<<<grid_size_hidden, block_size>>>(
        d_input, d_W1, d_b1, d_hidden, input_size, hidden_size);
    CHECK_CUDA(cudaGetLastError());

    // Kernel Capa Salida
    int grid_size_output = (output_size + block_size - 1) / block_size;
    forward_output_kernel<<<grid_size_output, block_size>>>(
        d_hidden, d_W2, d_b2, d_output, hidden_size, output_size);
    CHECK_CUDA(cudaGetLastError());

    // Retornar resultados a CPU
    CHECK_CUDA(cudaMemcpy(h_hidden, d_hidden, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> hidden_vec(h_hidden, h_hidden + hidden_size);
    std::vector<double> output_vec(h_output, h_output + output_size);

    return {hidden_vec, output_vec};
}
```

Ejecuta la inferencia de la red.

1. Transfiere los datos de entrada de la CPU a la GPU.
2. Lanza los kernels CUDA (`forward_hidden_kernel` y `forward_output_kernel`) que realizan las multiplicaciones matriciales y aplican las funciones de activación en paralelo.
3. Devuelve los resultados a la CPU para que el agente pueda tomar decisiones.

#### Función de Entrenamiento (Train Step)

```cpp
inline void CudaNeuralNet::train_step(const std::vector<double>& input, const std::vector<double>& target) {
    // Copiar datos a GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target, target.data(), output_size * sizeof(double), cudaMemcpyHostToDevice));

    // Forward Pass (similar a la función forward)
    // ... kernels forward ...

    // Backpropagation: Calcular Deltas
    compute_output_delta_kernel<<<grid_size_output, block_size>>>(
        d_output, d_target, d_output_delta, output_size);

    compute_hidden_delta_kernel<<<grid_size_hidden, block_size>>>(
        d_hidden, d_output_delta, d_W2, d_hidden_delta, hidden_size, output_size);

    // Actualización de Pesos (Gradient Descent)
    update_W2_kernel<<<grid_size_W2, block_size>>>(
        d_W2, d_b2, d_hidden, d_output_delta, learning_rate, hidden_size, output_size);

    update_W1_kernel<<<grid_size_W1, block_size>>>(
        d_W1, d_b1, d_input, d_hidden_delta, learning_rate, input_size, hidden_size);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Sincronizar pesos actualizados a CPU
    CHECK_CUDA(cudaMemcpy(h_W1, d_W1, input_size * hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, d_W2, hidden_size * output_size * sizeof(double), cudaMemcpyDeviceToHost));
}
```

Realiza un paso de descenso de gradiente estocástico (SGD) completamente en la GPU.

1. Calcula el error entre la salida de la red y el valor objetivo (Target Q-Value).
2. Calcula los gradientes para cada capa (`compute_output_delta`, `compute_hidden_delta`).
3. Actualiza los pesos y sesgos directamente en la memoria de la GPU para maximizar la eficiencia.

### 2.4. Agente Inteligente (`dqn_agent.h`)

Implementa la lógica de alto nivel del Double DQN y el Experience Replay.

#### Clase CudaDoubleDQNAgent

```cpp
class CudaDoubleDQNAgent {
public:
    CudaNeuralNet policy_net; // Red Online
    CudaNeuralNet target_net; // Red Target
    // ... parámetros (gamma, epsilon, memory) ...

    void remember(std::vector<double> s, int a, double r, std::vector<double> ns, bool d) {
        if (memory.size() >= max_memory) memory.pop_front();
        memory.push_back({s, a, r, ns, d});
    }

    void replay() {
        if (memory.size() < batch_size) return;

        std::vector<Transition> batch;
        // Muestreo aleatorio del buffer
        for(int i = 0; i < batch_size; i++) {
            int idx = (int)random_double(0, memory.size() - 1);
            batch.push_back(memory[idx]);
        }

        for (const auto& t : batch) {
            // 1. Obtener predicción actual
            auto fwd = policy_net.forward(t.state);
            std::vector<double> target_vector = fwd.second;

            double q_update = t.reward;

            if (!t.done) {
                // --- LOGICA DOUBLE DQN ---
                // A. Policy Net elige mejor accion en S_next
                int best_action_next = policy_net.forward_argmax(t.next_state);

                // B. Target Net evalua Q-value de esa accion
                double q_value_target = target_net.get_q_value(t.next_state, best_action_next);

                // C. Ecuacion de Bellman
                q_update += gamma * q_value_target;
            }

            target_vector[t.action] = q_update;
            policy_net.train_step(t.state, target_vector);
        }

        // Decaimiento de epsilon y actualización de Target Network
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;
        step_count++;
        if (step_count % target_update_freq == 0) {
            update_target_network();
        }
    }
};

```

- **remember:** Almacena la transición en una cola de memoria (`std::deque`) para romper la correlación temporal entre muestras consecutivas.
- **replay:** Es el núcleo del aprendizaje.

1. Toma una muestra aleatoria (batch) de experiencias pasadas.
2. Aplica la lógica Double DQN explicada en la introducción.
3. Calcula el nuevo objetivo y entrena la red `policy_net`.
4. Periódicamente copia los pesos de `policy_net` a `target_net`.

### 2.5. Cliente WebSocket (`socket_utils.h`)

Implementación manual ("bare-metal") de un cliente WebSocket en C++ utilizando sockets POSIX, sin bibliotecas externas de alto nivel.

#### Clase WebSocketClient

```cpp
class WebSocketClient {
private:
    // ... variables de socket y dirección ...

    std::string encode_websocket_frame(const std::string& message, uint8_t opcode = 0x81) {
        size_t len = message.length();
        std::string frame;

        // Byte 1: FIN + opcode
        frame.push_back(0x80 | opcode);

        // Byte 2: MASK bit + payload length
        if (len <= 125) {
            frame.push_back(0x80 | (len & 0x7F));
        } else if (len <= 65535) {
            // Lógica para payloads medianos
            frame.push_back(0x80 | 126);
            frame.push_back((len >> 8) & 0xFF);
            frame.push_back(len & 0xFF);
        }

        // Máscara (4 bytes) y enmascaramiento XOR
        uint32_t mask = 0x12345678;
        // ... push mask bytes ...

        if (len > 0) {
            for (size_t i = 0; i < len; i++) {
                char masked_char = message[i] ^ ((mask >> (8 * (3 - (i % 4)))) & 0xFF);
                frame.push_back(masked_char);
            }
        }
        return frame;
    }

public:
    bool connect_to_server() {
        // Creación del socket TCP
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        // ... configuración sockaddr_in ...
        connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

        // Handshake HTTP Upgrade
        send_websocket_handshake();

        // Verificación de respuesta HTTP 101 Switching Protocols
    }

    bool send_action(const std::string& action, int step, double reward) {
        std::string message = create_json_message(action, step, reward);
        std::string frame = encode_websocket_frame(message, 0x81);
        send(sockfd, frame.c_str(), frame.length(), 0);
        // ...
    }
};
```

Esta clase maneja la comunicación de bajo nivel.

- **connect_to_server:** Establece una conexión TCP estándar y luego realiza el "handshake" HTTP enviando las cabeceras `Upgrade: websocket` y `Sec-WebSocket-Key` requeridas por el protocolo RFC 6455.
- **encode_websocket_frame:** Construye manualmente la trama binaria del WebSocket. Esto incluye el bit FIN, el opcode (0x81 para texto), la longitud de la carga útil y, crucialmente, la clave de máscara (Masking Key) que se aplica con una operación XOR a los datos, lo cual es obligatorio para clientes WebSocket.

---

## 3. Ejecutables Principales

El proyecto genera tres ejecutables distintos, cada uno con un propósito específico en el ciclo de vida del desarrollo del modelo.

### 3.1. Entrenamiento (`main.cu`)

Este es el punto de entrada para entrenar el modelo desde cero.

```cpp
int main() {
    int N = 10;
    GridEnvironment env(N);
    CudaDoubleDQNAgent agent(env.get_state_size(), 64, env.get_action_size());

    int episodes = 1000;

    for (int e = 0; e < episodes; ++e) {
        auto state = env.reset();
        // ...
        while (!done && steps < 50) {
            // Ciclo de interacción: Actuar -> Entorno -> Recordar -> Entrenar
            int action = agent.act(state);
            auto res = env.step(action);
            agent.remember(state, action, res.first, next_state, res.second);
            agent.replay(); // Entrenamiento en GPU
        }
    }
    // Guardar modelo al finalizar
    agent.save("double_dqn_model");
    return 0;
}

```

Instancia el entorno y el agente, ejecuta 1000 episodios de entrenamiento donde el agente explora (epsilon-greedy) y aprende. Al final, serializa los pesos de la red neuronal a archivos binarios (`.bin`) para su uso posterior.

### 3.2. Test Local (`test_model.cu`)

Carga un modelo previamente entrenado y evalúa su rendimiento en la consola local.

```cpp
int main(int argc, char* argv[]) {
    // ...
    CudaDoubleDQNAgent agent(model_path);
    agent.epsilon = 0.0; // Desactivar exploración (solo explotación)

    // Ejecutar episodio de prueba
    while (!done && steps < 30) {
        int action = agent.act(state);
        auto res = env.step(action);
        // Mostrar progreso en consola
        std::cout << "Paso " << (steps + 1) << ": " << action_names[action] << std::endl;
    }
}
```

Validar que el modelo ha aprendido correctamente. Establece `epsilon = 0` para que el agente siempre elija la mejor acción conocida según la red neuronal, sin aleatoriedad.

### 3.3. Test con WebSocket (`test_socket.cu`)

Integra el modelo entrenado con el servidor de sockets documentado anteriormente.

```cpp
int main(int argc, char* argv[]) {
    // ... Carga modelo ...
    WebSocketClient ws_client(ws_address, ws_port);
    ws_client.connect_to_server();

    // Ciclo de inferencia
    while (!done) {
        int action = agent.act(state);
        auto res = env.step(action);

        // Enviar decisión al servidor Node.js
        if (ws_connected) {
            ws_client.send_action(action_name, steps + 1, res.first);
        }
    }
}
```

Actúa como el controlador del sistema "Robot/IA". En cada paso, calcula la acción óptima usando CUDA y la envía inmediatamente al servidor WebSocket. Esto permite visualizar el comportamiento del agente en una interfaz externa o controlar hardware remoto en tiempo real.

---

## 4. Compilación y Ejecución

Para compilar el proyecto se utiliza **CMake**, que gestiona las dependencias de CUDA y la compilación cruzada de código host/device.

### Comandos de Compilación

1. Crear directorio de construcción para mantener limpio el código fuente

```bash
mkdir build
cd build
```

2. Configurar el proyecto con CMake (detecta compilador NVCC y dependencias)

```bash
cmake ..
```

3. Compilar los ejecutables (utiliza makefiles generados)

```bash
make
```

### Ejecución de los Módulos

**1. Entrenar el modelo:**
Genera los archivos de pesos (`_policy.bin`, `_target.bin`) y parámetros.

```bash
./double_dqn_train
```

**2. Probar el modelo en consola:**
Carga los archivos generados y muestra una simulación de texto.

```bash
./double_dqn_test
```

**3. Ejecutar el cliente conectado al servidor:**
Requiere que el servidor `socket-server` (Node.js) esté corriendo previamente. Conecta la IA al bus de comunicación.

```bash
# Uso: ./double_dqn_socket [modelo] [ip] [puerto]
./double_dqn_socket double_dqn_model 127.0.0.1 5555
```
