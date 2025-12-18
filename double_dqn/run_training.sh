#!/bin/bash
echo "=== ENTRENAMIENTO DOUBLE DQN ==="

mkdir -p build
cd build

# Limpiar solo si se solicita
if [ "$1" = "clean" ]; then
    echo "🧹 Limpiando build..."
    rm -rf *
    cmake ..
fi

make

echo "🚀 Iniciando entrenamiento..."
./double_dqn_train

echo "📁 Modelo guardado como 'double_dqn_model'"
echo ""
echo "Para probar sin WebSocket: ./double_dqn_test"
echo "Para probar con WebSocket: ./double_dqn_socket"
echo "O usar los scripts: ./run_socket_test.sh"

cd ..