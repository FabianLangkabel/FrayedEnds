set(MADNESS_DIR "/usr/local/lib/cmake/madness")
find_package(MADNESS CONFIG REQUIRED)

message(STATUS "Using MADNESS installation at: ${MADNESS_DIR}")

set(SOURCE_FILES
"main.cpp"
"integrals.cpp"
)

add_executable(IntegralsRef ${SOURCE_FILES})
target_link_libraries(IntegralsRef madness)