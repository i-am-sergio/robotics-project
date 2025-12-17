cmake_minimum_required(VERSION 3.18)
project(DoubleDQN_CUDA)

# Configurar CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Configuración de compilación
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Flags de compilación
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Archivos del proyecto
set(SOURCES
    src/main.cu  # Cambiado de .cpp a .cu
)

set(HEADERS
    include/cuda_utils.h
    include/neural_net.h
    include/dqn_agent.h
    include/environment.h
)

set(CUDA_SOURCES
    kernels/kernels.cu
)

# Ejecutable
add_executable(double_dqn ${SOURCES} ${HEADERS} ${CUDA_SOURCES})
target_include_directories(double_dqn PRIVATE include)
target_include_directories(double_dqn PRIVATE kernels)

# Configurar CUDA
set_target_properties(double_dqn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Dependencias
find_package(Threads REQUIRED)
target_link_libraries(double_dqn Threads::Threads)