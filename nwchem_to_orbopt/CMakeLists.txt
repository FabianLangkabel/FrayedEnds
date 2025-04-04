set(MADNESS_DIR "/usr/local/madness/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)

find_package(MADNESS CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"converter.cpp"
)

add_executable(nwchem_to_orbopt ${SOURCE_FILES})
target_link_libraries(nwchem_to_orbopt madness)