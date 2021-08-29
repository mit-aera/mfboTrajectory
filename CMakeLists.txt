  
CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
project(pyMulticopterSim)

#### Options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

#### libs
set(INCLUDED_LIBS_DIR ${PROJECT_SOURCE_DIR}/libs)
get_filename_component(INCLUDED_LIBS_DIR ${INCLUDED_LIBS_DIR} ABSOLUTE)
include_directories(${INCLUDED_LIBS_DIR})

# pybind11
find_package(pybind11 QUIET COMPONENTS)
if (NOT pybind11_FOUND)
  unset(PYTHON_EXECUTABLE)
  find_package(PythonInterp)
  set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION_STRING})
  message(STATUS "[pyMulticopterSim] pybind11 NOT FOUND: ${PYBIND11_VERSION}")
  message(STATUS "[pyMulticopterSim] Using Python executable: ${PYTHON_EXECUTABLE}")
  message(STATUS "[pyMulticopterSim] Using Python version: ${PYBIND11_PYTHON_VERSION}")
  add_subdirectory(${INCLUDED_LIBS_DIR}/pybind11)
  set(PYBIND11_CPP_STANDARD -std=c++11)
  set(pybind11_FOUND true)
elseif(NOT pybind11_FOUND)
  message(STATUS "[pyMulticopterSim] FOUND pybind11: ${PYBIND11_VERSION}")
endif(NOT pybind11_FOUND)

# YAML
find_package(yaml-cpp QUIET COMPONENTS)
if ((NOT yaml-cpp_FOUND) AND (NOT YAML_CPP_FOUND))
  set(BUILD_SHARED_LIBS ON)
  set(BUILD_TESTS OFF)
  set(BUILD_EXAMPLES OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  add_subdirectory(${INCLUDED_LIBS_DIR}/yaml-cpp)
  include_directories(${INCLUDED_LIBS_DIR}/yaml-cpp/include/)
  set(YAML_CPP_FOUND true)
elseif(NOT YAML_CPP_FOUND)
  message(STATUS "[pyMulticopterSim] FOUND YAML_CPP: ${YAML_CPP_VERSION}")
endif(NOT YAML_CPP_FOUND)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR}/unsupported)

# multicopterDynamicsSim
include_directories(${INCLUDED_LIBS_DIR}/multicopterDynamicsSim)
add_library(inertialMeasurementSim ${INCLUDED_LIBS_DIR}/multicopterDynamicsSim/inertialMeasurementSim.cpp)
add_library(multicopterDynamicsSim ${INCLUDED_LIBS_DIR}/multicopterDynamicsSim/multicopterDynamicsSim.cpp)
target_link_libraries(multicopterDynamicsSim inertialMeasurementSim)

# python modules
pybind11_add_module(multicopter_dynamics_sim SHARED
  ${INCLUDED_LIBS_DIR}/pyMulticopterDynamicsSim/pyMulticopterDynamicsSim.cpp
)
target_link_libraries(multicopter_dynamics_sim PRIVATE 
  multicopterDynamicsSim
)
