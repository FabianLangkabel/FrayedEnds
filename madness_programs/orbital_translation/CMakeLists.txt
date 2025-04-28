set(CMAKE_CXX_STANDARD 17)
set(MADNESS_DIR "/usr/local/madness/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"translator.cpp"
)

add_executable(OrbitalTranslation ${SOURCE_FILES})
target_link_libraries(OrbitalTranslation madness)