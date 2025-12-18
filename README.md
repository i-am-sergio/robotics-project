## Socket server

### Inicializar proyecto
```sh
mkdir websocket-dqn-server
cd websocket-dqn-server
npm init -y
```

### Instalar WebSocket
```sh
npm install ws
```

### Ejecutar servidor
```sh
node server.js
```


## Instructions Sender

### Compilar versión simple (Linux/Mac)
```sh
g++ -std=c++11 -o sender sender.cpp
```

### Ejecutar
```sh
./sender
```