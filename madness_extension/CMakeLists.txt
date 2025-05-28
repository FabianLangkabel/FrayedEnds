set(MADNESS_DIR "/usr/local/lib/cmake/madness")
include_directories(
    "${CMAKE_SOURCE_DIR}/eigen"
)
find_package(MADNESS CONFIG REQUIRED)

message(STATUS "MadPy using MADNESS installation at: ${MADNESS_DIR}")


set(DEV_MODULE Development.Module)

# TODO: adapt for other venvs and dev container
# Ensure Python is found explicitly within the Conda environment
if (DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
else()
  message(FATAL_ERROR "Conda environment not detected. Please activate your Conda environment.")
endif()
find_package(Python 3.11 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
if (NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Python 3.11 not found in the Conda environment.")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


set(SOURCE_FILES
"main.cpp"
"optimization.cpp"
"functionsaver.hpp"
"pno_interface.hpp"
"sum_of_gaussians.hpp"
"MadnessProcess.hpp"
"Functor_test.hpp"
)

nanobind_add_module(MadPy ${SOURCE_FILES})
target_link_libraries(MadPy PRIVATE madness)

