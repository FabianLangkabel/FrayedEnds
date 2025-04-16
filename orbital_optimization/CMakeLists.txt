set(MADNESS_DIR "/usr/local/madness/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"optimization.cpp"
)

add_executable(OrbitalOptimization ${SOURCE_FILES})
target_link_libraries(OrbitalOptimization madness)