#!/bin/bash
echo "=== PRUEBA DE MODELO CON WEBSOCKET ==="

# Construir si es necesario
if [ ! -d "build" ]; then
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..
fi

cd build

# Ejecutar con parámetros opcionales
# Uso: ./run_socket_test.sh [modelo] [ip] [puerto]
MODEL_PATH="${1:-double_dqn_model}"
WS_ADDRESS="${2:-127.0.0.1}"
WS_PORT="${3:-5555}"

echo "Modelo: $MODEL_PATH"
echo "WebSocket: ws://$WS_ADDRESS:$WS_PORT"

# Verificar que el modelo existe
# if [ ! -f "../${MODEL_PATH}_policy.bin" ]; then
#     echo "❌ Error: No se encontró el modelo $MODEL_PATH"
#     echo "Primero ejecuta el entrenamiento: ./run_training.sh"
#     exit 1
# fi

# Copiar modelo al directorio build si no está ahí
if [ ! -f "${MODEL_PATH}_policy.bin" ]; then
    echo "📋 Copiando modelo al directorio build..."
    cp ../${MODEL_PATH}_* .
fi

# Ejecutar
echo "🚀 Iniciando prueba con WebSocket..."
./double_dqn_socket "$MODEL_PATH" "$WS_ADDRESS" "$WS_PORT"

cd ..