cmake_minimum_required(VERSION 3.10)
project(DoubleDQN_CUDA)

# Configurar CUDA
enable_language(CUDA)

# Configuración de compilación
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Flags de compilación
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Archivos del proyecto
set(HEADERS
    include/cuda_utils.h
    include/neural_net.h
    include/dqn_agent.h
    include/environment.h
    include/socket_utils.h 
)

set(CUDA_SOURCES
    kernels/kernels.cu
)

# Ejecutable de entrenamiento
add_executable(double_dqn_train src/main.cu ${HEADERS} ${CUDA_SOURCES})
target_include_directories(double_dqn_train PRIVATE include)
target_include_directories(double_dqn_train PRIVATE kernels)
set_target_properties(double_dqn_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70"
)

# Ejecutable de test
add_executable(double_dqn_test src/test_model.cu ${HEADERS} ${CUDA_SOURCES})
target_include_directories(double_dqn_test PRIVATE include)
target_include_directories(double_dqn_test PRIVATE kernels)
set_target_properties(double_dqn_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70"
)


# Ejecutable de test con WebSocket
add_executable(double_dqn_socket src/test_socket.cu ${HEADERS} ${CUDA_SOURCES})
target_include_directories(double_dqn_socket PRIVATE include)
target_include_directories(double_dqn_socket PRIVATE kernels)
set_target_properties(double_dqn_socket PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70"
)

# Dependencias
find_package(Threads REQUIRED)
target_link_libraries(double_dqn_train Threads::Threads)
target_link_libraries(double_dqn_test Threads::Threads)
target_link_libraries(double_dqn_socket Threads::Threads)