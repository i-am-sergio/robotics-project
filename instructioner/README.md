# Servicio: Instructions Sender

## 1. Información General

### Tecnologías Utilizadas

Este servicio está desarrollado en **C++** estándar, utilizando la API de **Sockets POSIX** (Portable Operating System Interface) para la comunicación en red. No depende de bibliotecas de terceros (como Boost o WebSocket++), implementando el protocolo WebSocket de manera nativa y "bare-metal" mediante la manipulación directa de bytes y cabeceras TCP/IP.

### Dependencias

El código utiliza únicamente bibliotecas estándar del sistema y de C++:

- `<sys/socket.h>`, `<netinet/in.h>`, `<arpa/inet.h>`: Para la creación y gestión de sockets de red y direcciones IP.

- `<unistd.h>`: Para operaciones del sistema como `close()` y `usleep()`.

- `<iostream>`, `<string>`, `<cstring>`: Para entrada/salida estándar y manipulación de cadenas de texto.

### Propósito General

El `Instructions Sender` actúa como un cliente de diagnóstico y prueba unitaria para el servidor WebSocket. Su propósito es establecer una conexión manual, realizar el protocolo de enlace (handshake) y enviar un comando de control predefinido ("Arriba") codificado en JSON, validando así la capacidad del servidor para procesar tramas enmascaradas y responder adecuadamente.

### Objetivo Logrado

Este módulo permite aislar la lógica de comunicación de la lógica de inteligencia artificial. Logra:

1. Verificar la conectividad TCP con el puerto 5555 del host local.

2. Validar la implementación del protocolo de "Upgrade" de HTTP a WebSocket.

3. Demostrar la codificación correcta de tramas WebSocket (incluyendo el bit de máscara obligatorio para clientes) sin la sobrecarga de un framework completo.

---

## 2. Análisis Técnico y Descripción de Funciones

A continuación se detalla la implementación del archivo `sender.cpp`.

### Generación de Carga Útil (Payload)

**Función:** `create_json_message`

```cpp
std::string create_json_message(const std::string& command) {
    return "{\"command\":\"" + command + "\"}";
}
```

Esta función auxiliar toma una cadena de texto que representa una instrucción (por ejemplo, "Arriba") y la encapsula manualmente en una estructura JSON válida. El resultado es una cadena con el formato `{"command":"Arriba"}`, que es el formato esperado por el servidor para procesar instrucciones.

### Protocolo de Enlace (Handshake)

**Función:** `send_websocket_handshake`

```cpp
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
```

Esta función construye y envía la solicitud HTTP inicial requerida para elevar la conexión a WebSocket (RFC 6455). Envía las cabeceras críticas `Upgrade: websocket` y `Connection: Upgrade`, junto con una clave base64 estática (`Sec-WebSocket-Key`) y la versión del protocolo. Retorna `true` si el envío de datos al socket fue exitoso.

### Codificación de Tramas WebSocket

**Función:** `encode_websocket_frame`

```cpp
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
```

Implementa la lógica de bajo nivel para empaquetar datos según el estándar WebSocket para clientes:

1.  **Byte 1:** Establece el bit `FIN` (0x80) indicando que es el final del mensaje y añade el `opcode` (0x81 para texto, 0x88 para cierre).

2.  **Byte 2 y Longitud:** Establece el bit de máscara (0x80, obligatorio para clientes que envían al servidor) y codifica la longitud del mensaje. Maneja longitudes pequeñas (<= 125 bytes) y medianas (<= 65535 bytes) ajustando los bytes subsiguientes.

3.  **Enmascaramiento:** Genera una clave de máscara de 4 bytes (en este caso estática `0x12345678` para consistencia en pruebas) y aplica una operación XOR byte a byte sobre el mensaje original. Esto es fundamental para evitar problemas de caché en proxies intermedios.

### Función Principal y Flujo de Ejecución

**Función:** `main`

```cpp
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
    usleep(100000);
    close(sockfd);
    std::cout << "👋 Connection closed" << std::endl;
    return 0;
}
```

El bloque principal orquesta el ciclo de vida del cliente:

1.  **Inicialización:** Crea un socket TCP (`AF_INET`, `SOCK_STREAM`) y define la dirección de conexión como `127.0.0.1:5555`.
2.  **Conexión:** Establece la conexión física con el servidor.
3.  **Handshake:** Invoca `send_websocket_handshake` y entra en un bucle de espera activa (polling) hasta recibir la confirmación `HTTP/1.1 101` del servidor, indicando que el protocolo ha cambiado a WebSocket.
4.  **Transmisión:** Construye un mensaje JSON con el comando "Arriba", lo codifica en una trama binaria y lo envía por el socket.
5.  **Recepción:** Espera brevemente cualquier eco o confirmación del servidor.
6.  **Cierre:** Envía explícitamente una trama de control de cierre (Opcode `0x88`), espera un breve periodo para asegurar la transmisión y cierra el descriptor del socket.

---

## 3. Salida de Ejecución

A continuación se presenta la salida en consola esperada al ejecutar este cliente mientras el servidor está activo.

```text
🚀 Starting WebSocket client
📡 Connecting to ws://localhost:5555
✅ Socket connected, sending handshake...
📤 Handshake sent, waiting for response...
✅ WebSocket connection established
📤 Sending single command...
✅ Sent: {"command":"Arriba"}
⏳ Waiting for response...
📥 Received 127 bytes from server
🔌 Sending close frame...
👋 Connection closed
```

---

## 4. Compilación y Ejecución

Dado que este archivo es independiente de la lógica CUDA del proyecto principal, se puede compilar utilizando `g++` estándar.

### Comandos

1. Compilar el cliente emisor

```bash
g++ -std=c++11 -o sender sender.cpp
```

2. Ejecutar el cliente (Asegurarse de que el servidor esté corriendo primero)

```bash
./sender
```

### Funciones del Ejecutable

- **`./sender`**: Inicia el proceso cliente que intenta conectarse inmediatamente a `localhost:5555`, realizar el handshake, enviar el comando "Arriba" y cerrar la conexión. Es útil para verificar rápidamente si el servidor está aceptando conexiones y decodificando mensajes correctamente sin necesidad de cargar los modelos de IA.
