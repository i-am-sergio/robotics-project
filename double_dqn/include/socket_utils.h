#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <mutex>

class WebSocketClient {
private:
    int sockfd = -1;
    bool connected = false;
    std::mutex socket_mutex;
    std::string server_address = "10.7.134.228"; // "127.0.0.1";
    int port = 5555;
    
    std::string create_json_message(const std::string& command, int step, double reward) {
        return "{\"command\":\"" + command + "\",\"step\":" + 
               std::to_string(step) + ",\"reward\":" + 
               std::to_string(reward) + "}";
    }
    
    bool send_websocket_handshake() {
        std::string handshake = 
            "GET / HTTP/1.1\r\n"
            "Host: " + server_address + ":" + std::to_string(port) + "\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n";
        
        std::lock_guard<std::mutex> lock(socket_mutex);
        return send(sockfd, handshake.c_str(), handshake.length(), 0) > 0;
    }
    
    std::string encode_websocket_frame(const std::string& message, uint8_t opcode = 0x81) {
        size_t len = message.length();
        std::string frame;
        
        // Byte 1: FIN + opcode
        frame.push_back(0x80 | opcode);
        
        // Byte 2: MASK bit + payload length
        if (len <= 125) {
            frame.push_back(0x80 | (len & 0x7F));
        } else if (len <= 65535) {
            frame.push_back(0x80 | 126);
            frame.push_back((len >> 8) & 0xFF);
            frame.push_back(len & 0xFF);
        }
        
        // Máscara (4 bytes)
        uint32_t mask = 0x12345678;
        frame.push_back((mask >> 24) & 0xFF);
        frame.push_back((mask >> 16) & 0xFF);
        frame.push_back((mask >> 8) & 0xFF);
        frame.push_back(mask & 0xFF);
        
        // Aplicar máscara al payload
        if (len > 0) {
            for (size_t i = 0; i < len; i++) {
                char masked_char = message[i] ^ ((mask >> (8 * (3 - (i % 4)))) & 0xFF);
                frame.push_back(masked_char);
            }
        }
        
        return frame;
    }
    
public:
    WebSocketClient() = default;
    
    WebSocketClient(const std::string& addr, int p) : server_address(addr), port(p) {}
    
    bool connect_to_server() {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::cerr << "❌ Error creating socket" << std::endl;
            return false;
        }
        
        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, server_address.c_str(), &serv_addr.sin_addr) <= 0) {
            std::cerr << "❌ Invalid address: " << server_address << std::endl;
            close(sockfd);
            sockfd = -1;
            return false;
        }
        
        std::cout << "📡 Connecting to ws://" << server_address << ":" << port << "..." << std::endl;
        
        if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "❌ Connection failed" << std::endl;
            close(sockfd);
            sockfd = -1;
            return false;
        }
        
        std::cout << "✅ Socket connected, sending handshake..." << std::endl;
        
        if (!send_websocket_handshake()) {
            std::cerr << "❌ Handshake failed" << std::endl;
            close(sockfd);
            sockfd = -1;
            return false;
        }
        
        // Esperar respuesta del handshake
        char buffer[1024];
        bool handshake_complete = false;
        
        for (int i = 0; i < 50 && !handshake_complete; i++) {
            int bytes = recv(sockfd, buffer, sizeof(buffer) - 1, MSG_DONTWAIT);
            if (bytes > 0) {
                buffer[bytes] = '\0';
                std::string response(buffer);
                
                if (response.find("HTTP/1.1 101") != std::string::npos ||
                    response.find("Switching Protocols") != std::string::npos) {
                    std::cout << "✅ WebSocket connection established" << std::endl;
                    handshake_complete = true;
                    connected = true;
                }
            }
            usleep(100000);
        }
        
        if (!handshake_complete) {
            std::cerr << "❌ Handshake timeout" << std::endl;
            close(sockfd);
            sockfd = -1;
            return false;
        }
        
        return true;
    }
    
    bool send_action(const std::string& action, int step, double reward) {
        if (!connected || sockfd < 0) {
            std::cerr << "❌ Not connected to WebSocket server" << std::endl;
            return false;
        }
        
        std::string message = create_json_message(action, step, reward);
        std::string frame = encode_websocket_frame(message, 0x81);
        
        std::lock_guard<std::mutex> lock(socket_mutex);
        int sent = send(sockfd, frame.c_str(), frame.length(), 0);
        
        if (sent > 0) {
            std::cout << "📤 Sent to server: " << action << " (step " << step << ", reward " << reward << ")" << std::endl;
            return true;
        } else {
            std::cerr << "❌ Error sending message" << std::endl;
            connected = false;
            return false;
        }
    }
    
    void disconnect() {
        if (sockfd >= 0) {
            std::lock_guard<std::mutex> lock(socket_mutex);
            
            if (connected) {
                // Enviar frame de cierre
                std::string close_frame = encode_websocket_frame("", 0x88);
                send(sockfd, close_frame.c_str(), close_frame.length(), 0);
                usleep(100000);
            }
            
            close(sockfd);
            sockfd = -1;
            connected = false;
            std::cout << "👋 WebSocket connection closed" << std::endl;
        }
    }
    
    bool is_connected() const {
        return connected;
    }
    
    ~WebSocketClient() {
        disconnect();
    }
};

#endif // SOCKET_UTILS_H