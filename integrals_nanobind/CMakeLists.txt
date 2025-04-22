set(MADNESS_DIR "/usr/local/lib/cmake/madness")
find_package(MADNESS CONFIG REQUIRED)

message(STATUS "Using MADNESS installation at: ${MADNESS_DIR}")

set(DEV_MODULE Development.Module)
find_package(Python 3.11 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(SOURCE_FILES
"main.cpp"
"integrals.cpp"
)

nanobind_add_module(MRA_integrals ${SOURCE_FILES})
target_link_libraries(MRA_integrals PRIVATE madness)