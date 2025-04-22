set(MADNESS_DIR "/usr/local/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

message(STATUS "Using MADNESS installation at: ${MADNESS_DIR}")

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"optimization.cpp"
)

add_executable(OrbitalOptimization ${SOURCE_FILES})
target_link_libraries(OrbitalOptimization madness)