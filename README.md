## Socket server

### Inicializar proyecto
mkdir websocket-dqn-server
cd websocket-dqn-server
npm init -y

### Instalar WebSocket
npm install ws

### Ejecutar servidor
node server.js



## Instructions Sender

### Compilar versión simple (Linux/Mac)
g++ -std=c++11 -o simple_sender simple_sender.cpp
g++ -std=c++11 -o simple_sender simple_sender.cpp -lwebsockets -lpthread

### Ejecutar
./simple_sender