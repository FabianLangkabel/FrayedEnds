cmake_minimum_required(VERSION 3.22)
project(OrbitalOptimization)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(orbital_optimization)
add_subdirectory(integrals_ref)
add_subdirectory(integrals_nanobind)
add_subdirectory(nwchem_to_orbopt)