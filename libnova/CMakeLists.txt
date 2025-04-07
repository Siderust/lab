cmake_minimum_required(VERSION 3.10)
project(libnova_metrics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluye encabezados desde libnova
include_directories(/usr/local/include)

add_executable(precission precission.cpp)
target_link_libraries(precission nova m)

add_executable(performance performance.cpp)
target_link_libraries(performance nova m)