cmake_minimum_required(VERSION 3.22)
project(OrbitalOptimization)
set(CMAKE_CXX_STANDARD 17)
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"orbital_optimization/main.cpp"
"orbital_optimization/optimization.cpp"
)

add_executable(OrbitalOptimization ${SOURCE_FILES})
target_link_libraries(OrbitalOptimization madness)