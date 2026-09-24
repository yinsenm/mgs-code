cmake_minimum_required(VERSION 3.15)
project(cgerber LANGUAGES CXX)

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG QUIET)

pybind11_add_module(cgerber cgerber.cpp)
set_target_properties(cgerber PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

if(Eigen3_FOUND)
    target_link_libraries(cgerber PRIVATE Eigen3::Eigen)
else()
    set(EIGEN3_INCLUDE_DIR "/opt/homebrew/include/eigen3" CACHE PATH "Path to Eigen3 headers")
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen")
        message(FATAL_ERROR "Eigen3 headers not found. Set -DEIGEN3_INCLUDE_DIR=/path/to/eigen3")
    endif()
    target_include_directories(cgerber PRIVATE "${EIGEN3_INCLUDE_DIR}")
endif()
