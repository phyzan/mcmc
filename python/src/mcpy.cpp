#include "include/mcpy/mcpy.hpp"

PYBIND11_MODULE(mcpy, m)
{
    define_base_module<double>(m);
}


//clang++ -O3 -Wall -Wextra -Wpedantic -march=x86-64 -shared -std=c++20 -fvisibility=hidden -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) python/mcpy.cpp -o python/mcpy$(python3-config --extension-suffix)