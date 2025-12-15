#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <libwebsockets.h>

std::string get_random_command() {
    static const std::string commands[] = {"Arriba", "Abajo", "Izquierda", "Derecha"};
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 3);
    
    return commands[dis(gen)];
}

std::string create_json_message(const std::string& command, int step) {
    return "{\"command\":\"" + command + 
           "\",\"step\":" + std::to_string(step) + 
           ",\"source\":\"c++_libwebsockets_client\"}";
}

// Variables globales para controlar el flujo
static std::atomic<int> message_count(0);
static std::atomic<int> current_step(0);
static std::atomic<bool> connected(false);
static struct lws *global_wsi = nullptr;

static int callback(struct lws *wsi, enum lws_callback_reasons reason, 
                    void *user, void *in, size_t len) {
    switch(reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            std::cout << "✅ WebSocket connection established" << std::endl;
            connected = true;
            global_wsi = wsi;
            // Solicitar escritura inmediata
            lws_callback_on_writable(wsi);
            break;
            
        case LWS_CALLBACK_CLIENT_RECEIVE:
            std::cout << "📥 Received: " << std::string((char*)in, len) << std::endl;
            break;
            
        case LWS_CALLBACK_CLIENT_WRITEABLE: {
            if (message_count < 20 && connected) {
                std::string command = get_random_command();
                std::string message = create_json_message(command, ++current_step);
                
                unsigned char *buffer = (unsigned char*)malloc(LWS_PRE + message.length() + 1);
                memcpy(buffer + LWS_PRE, message.c_str(), message.length());
                
                int n = lws_write(wsi, buffer + LWS_PRE, message.length(), LWS_WRITE_TEXT);
                free(buffer);
                
                if (n < 0) {
                    std::cerr << "❌ Error writing to WebSocket" << std::endl;
                    return -1;
                }
                
                std::cout << "📤 Sent: " << command << " (Step: " << current_step << ")" << std::endl;
                message_count++;
                
                // Programar próximo envío después de 1 segundo
                if (message_count < 20) {
                    // Usar un timer para próximo envío
                    lws_callback_on_writable(wsi);
                } else {
                    std::cout << "✅ All messages sent" << std::endl;
                }
            }
            break;
        }
            
        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            std::cerr << "❌ Connection error" << std::endl;
            connected = false;
            break;
            
        case LWS_CALLBACK_CLIENT_CLOSED:
            std::cout << "🔌 Connection closed" << std::endl;
            connected = false;
            break;
    }
    
    return 0;
}

int main() {
    struct lws_context_creation_info info;
    struct lws_client_connect_info connect_info;
    struct lws_context *context;
    struct lws_protocols protocols[] = {
        {
            "my-protocol",
            callback,
            0,
            1024,
        },
        { NULL, NULL, 0, 0 }
    };
    
    // Inicializar info
    memset(&info, 0, sizeof(info));
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;
    info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    
    // Crear contexto
    context = lws_create_context(&info);
    if (!context) {
        std::cerr << "❌ Error creating libwebsockets context" << std::endl;
        return 1;
    }
    
    // Configurar conexión
    memset(&connect_info, 0, sizeof(connect_info));
    connect_info.context = context;
    connect_info.address = "localhost";
    connect_info.port = 5555;
    connect_info.path = "/";
    connect_info.host = connect_info.address;
    connect_info.origin = connect_info.address;
    connect_info.protocol = protocols[0].name;
    connect_info.ietf_version_or_minus_one = -1;
    connect_info.client_exts = NULL;
    
    std::cout << "🚀 Starting libwebsockets client" << std::endl;
    std::cout << "📡 Connecting to ws://localhost:5555" << std::endl;
    
    // Conectar
    global_wsi = lws_client_connect_via_info(&connect_info);
    if (!global_wsi) {
        std::cerr << "❌ Error connecting to server" << std::endl;
        lws_context_destroy(context);
        return 1;
    }
    
    std::cout << "📤 Sending commands every 1 second... (Press Ctrl+C to stop)" << std::endl;
    
    // Bucle principal con tiempo controlado
    int max_messages = 20;
    auto start_time = std::chrono::steady_clock::now();
    
    while (message_count < max_messages) {
        // Ejecutar servicio con timeout
        lws_service(context, 100); // 100ms timeout para ser más responsivo
        
        // Controlar tiempo entre mensajes
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        
        if (elapsed >= 1000 && connected && message_count < max_messages) {
            // Solicitar escritura
            if (global_wsi) {
                lws_callback_on_writable(global_wsi);
            }
            start_time = now;
        }
        
        // Pequeña pausa para no saturar la CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Esperar un poco antes de cerrar
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Limpiar
    lws_context_destroy(context);
    std::cout << "\n👋 Client finished" << std::endl;
    
    return 0;
}