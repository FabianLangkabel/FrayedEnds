set(CMAKE_CXX_STANDARD 17)
set(MADNESS_DIR "/usr/local/madness/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"spinorb_optimizer.cpp"
)

add_executable(SpinorbOpt ${SOURCE_FILES})
target_link_libraries(SpinorbOpt madness)