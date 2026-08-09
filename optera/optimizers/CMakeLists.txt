cmake_minimum_required(VERSION 3.14)
project(OpteraOptimizerPSO VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(cpp/include)

add_executable(pso_benchmark
    cpp/src/main.cpp
    cpp/src/pso.cpp
    cpp/src/benchmark.cpp
)

add_library(optera_pso SHARED
    cpp/src/pso.cpp
    cpp/src/benchmark.cpp
)

target_include_directories(optera_pso PUBLIC cpp/include)
