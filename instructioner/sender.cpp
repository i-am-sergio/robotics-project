// sender.cpp
#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

std::string create_json_message(const std::string& command) {
    return "{\"command\":\"" + command + "\"}";
}

bool send_websocket_handshake(int sockfd) {
    std::string handshake = 
        "GET / HTTP/1.1\r\n"
        "Host: localhost:5555\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n";
    
    return send(sockfd, handshake.c_str(), handshake.length(), 0) > 0;
}

std::string encode_websocket_frame(const std::string& message, uint8_t opcode = 0x81) {
    size_t len = message.length();
    std::string frame;
    
    // Byte 1: FIN + opcode
    frame.push_back(0x80 | opcode);
    
    // Byte 2: MASK bit + payload length
    if (len <= 125) {
        frame.push_back(0x80 | (len & 0x7F)); // MASK bit + length
    } else if (len <= 65535) {
        frame.push_back(0x80 | 126); // MASK bit + length code 126
        frame.push_back((len >> 8) & 0xFF); // Length MSB
        frame.push_back(len & 0xFF); // Length LSB
    }
    
    // Máscara (4 bytes)
    uint32_t mask = 0x12345678; // Misma máscara para consistencia
    frame.push_back((mask >> 24) & 0xFF);
    frame.push_back((mask >> 16) & 0xFF);
    frame.push_back((mask >> 8) & 0xFF);
    frame.push_back(mask & 0xFF);
    
    // Aplicar máscara al payload (si hay payload)
    if (len > 0) {
        for (size_t i = 0; i < len; i++) {
            char masked_char = message[i] ^ ((mask >> (8 * (3 - (i % 4)))) & 0xFF);
            frame.push_back(masked_char);
        }
    }
    
    return frame;
}

int main() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "❌ Error creating socket" << std::endl;
        return 1;
    }
    
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(5555);
    
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "❌ Invalid address" << std::endl;
        close(sockfd);
        return 1;
    }
    
    std::cout << "🚀 Starting WebSocket client" << std::endl;
    std::cout << "📡 Connecting to ws://localhost:5555" << std::endl;
    
    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "❌ Connection failed" << std::endl;
        close(sockfd);
        return 1;
    }
    
    std::cout << "✅ Socket connected, sending handshake..." << std::endl;
    
    if (!send_websocket_handshake(sockfd)) {
        std::cerr << "❌ Handshake failed" << std::endl;
        close(sockfd);
        return 1;
    }
    
    std::cout << "📤 Handshake sent, waiting for response..." << std::endl;
    
    char buffer[1024];
    bool handshake_complete = false;
    
    // Esperar respuesta del handshake
    for (int i = 0; i < 50 && !handshake_complete; i++) {
        int bytes = recv(sockfd, buffer, sizeof(buffer) - 1, MSG_DONTWAIT);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            std::string response(buffer);
            
            if (response.find("HTTP/1.1 101") != std::string::npos) {
                std::cout << "✅ WebSocket connection established" << std::endl;
                handshake_complete = true;
            }
        }
        usleep(100000);
    }
    
    if (!handshake_complete) {
        std::cerr << "❌ Handshake timeout" << std::endl;
        close(sockfd);
        return 1;
    }
    
    std::cout << "📤 Sending single command..." << std::endl;
    
    // Crear y enviar el mensaje
    std::string message = create_json_message("Arriba");
    std::string frame = encode_websocket_frame(message, 0x81); // Text frame
    
    int sent = send(sockfd, frame.c_str(), frame.length(), 0);
    if (sent > 0) {
        std::cout << "✅ Sent: " << message << std::endl;
    } else {
        std::cerr << "❌ Error sending message" << std::endl;
        close(sockfd);
        return 1;
    }
    
    // Esperar breve respuesta
    std::cout << "⏳ Waiting for response..." << std::endl;
    for (int i = 0; i < 20; i++) {
        int bytes = recv(sockfd, buffer, sizeof(buffer) - 1, MSG_DONTWAIT);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            std::cout << "📥 Received " << bytes << " bytes from server" << std::endl;
            break;
        }
        usleep(100000);
    }
    
    // Enviar frame de cierre CORRECTAMENTE enmascarado
    std::cout << "🔌 Sending close frame..." << std::endl;
    std::string close_frame = encode_websocket_frame("", 0x88); // Close frame
    send(sockfd, close_frame.c_str(), close_frame.length(), 0);
    
    // Esperar un poco para que el servidor procese el cierre
    usleep(100000);
    
    close(sockfd);
    std::cout << "👋 Connection closed" << std::endl;
    
    return 0;
}