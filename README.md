## Compile and Run DDQN

```sh
git clone https://github.com/i-am-sergio/robotics-project/
cd robotics-project/double_dqn/
mkdir build && cd build && cmake .. && make
cp ../models/* .
./double_dqn_test
```

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