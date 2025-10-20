cmake_minimum_required(VERSION 3.15...3.27)
project(mcpy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create the Python extension module
pybind11_add_module(mcpy src/mcpy.cpp)

# Include directories
target_include_directories(mcpy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link OpenMP
target_link_libraries(mcpy PRIVATE OpenMP::OpenMP_CXX)

# Compiler flags
target_compile_options(mcpy PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
    -march=x86-64
    -fopenmp
)

# Set visibility
set_target_properties(mcpy PROPERTIES
    CXX_VISIBILITY_PRESET hidden
)

# Installation
install(TARGETS mcpy LIBRARY DESTINATION mcpy)
