set(MADNESS_DIR "/usr/local/madness/lib/cmake/madness")
find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"integrals.cpp"
)

add_executable(IntegralsRef ${SOURCE_FILES})
target_link_libraries(IntegralsRef madness)