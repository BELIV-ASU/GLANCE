################################################################################
# CMakeLists.txt
# ──────────────────────────────────────────────────────────────────────────────
# Project : Empirical Study of VLMs (7B) Performance on Edge
#           in Real-World Traffic Violations
#
# Build System for hybrid C++ / Python research project.
#
# Dependencies:
#   - ROS 2 (Humble+): rclcpp, sensor_msgs, std_msgs, rosbag2_cpp
#   - LibTorch          (C++ frontend for PyTorch, used for SNN prototyping)
#   - TensorRT          (NVIDIA high-perf inference)
#   - ONNX Runtime      (cross-platform model inference)
#   - CUDA / cuDNN      (GPU acceleration)
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
################################################################################
cmake_minimum_required(VERSION 3.22)
project(edge_vlm_study
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Edge VLM 7B Traffic-Violation Empirical Study"
)

# ── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build options ────────────────────────────────────────────────────────────
option(BUILD_TESTS        "Build unit tests"                     ON)
option(ENABLE_TENSORRT    "Enable TensorRT inference backend"    ON)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime inference backend" ON)
option(ENABLE_SNN         "Enable Spiking Neural Network runtime" ON)

# ── Compiler flags ───────────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
)

###############################################################################
# Find Dependencies
###############################################################################

# ── ROS 2 ────────────────────────────────────────────────────────────────────
# TODO: Ensure you have sourced your ROS 2 workspace:
#       source /opt/ros/humble/setup.bash
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# ── CUDA ─────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── LibTorch ─────────────────────────────────────────────────────────────────
# TODO: Set Torch_DIR to your LibTorch installation, e.g.:
#       cmake .. -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
find_package(Torch REQUIRED)

# ── TensorRT (optional) ─────────────────────────────────────────────────────
if(ENABLE_TENSORRT)
    # TODO: Set TensorRT_DIR or CMAKE_PREFIX_PATH to your TensorRT install
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu
              /usr/local/TensorRT/include
    )
    find_library(TENSORRT_LIB nvinfer
        HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu
              /usr/local/TensorRT/lib
    )
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
        message(STATUS "Found TensorRT: ${TENSORRT_LIB}")
        set(TENSORRT_FOUND TRUE)
    else()
        message(WARNING "TensorRT NOT found – TRT backend will be stubbed out.")
        set(TENSORRT_FOUND FALSE)
    endif()
endif()

# ── ONNX Runtime (optional) ─────────────────────────────────────────────────
if(ENABLE_ONNXRUNTIME)
    # TODO: Set ONNXRUNTIME_ROOT to your ONNX Runtime install directory
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include
              /usr/local/include/onnxruntime
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib
              /usr/local/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        set(ONNXRUNTIME_FOUND TRUE)
    else()
        message(WARNING "ONNX Runtime NOT found – ORT backend will be stubbed out.")
        set(ONNXRUNTIME_FOUND FALSE)
    endif()
endif()

###############################################################################
# Include directories
###############################################################################
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

if(ONNXRUNTIME_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()

###############################################################################
# Source files
###############################################################################
set(PROJECT_SOURCES
    src/DataIngestionNode.cpp
    src/VLMInferenceEngine.cpp
    src/SNNRuntimeInterface.cpp
    src/EdgeEvaluator.cpp
)

###############################################################################
# Library target (shared among nodes)
###############################################################################
add_library(${PROJECT_NAME}_lib SHARED ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME}_lib
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# ROS 2 dependencies
ament_target_dependencies(${PROJECT_NAME}_lib
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    rosbag2_cpp
)

# Conditional backend linkage
if(TENSORRT_FOUND)
    target_link_libraries(${PROJECT_NAME}_lib ${TENSORRT_LIB})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC ENABLE_TENSORRT=1)
endif()

if(ONNXRUNTIME_FOUND)
    target_link_libraries(${PROJECT_NAME}_lib ${ONNXRUNTIME_LIB})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC ENABLE_ONNXRUNTIME=1)
endif()

if(ENABLE_SNN)
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC ENABLE_SNN=1)
endif()

###############################################################################
# Executable: Main edge deployment node
###############################################################################
add_executable(edge_vlm_node src/main.cpp)
target_link_libraries(edge_vlm_node ${PROJECT_NAME}_lib)
ament_target_dependencies(edge_vlm_node rclcpp)

###############################################################################
# Install
###############################################################################
install(TARGETS edge_vlm_node ${PROJECT_NAME}_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

###############################################################################
# Tests
###############################################################################
if(BUILD_TESTS)
    # TODO: Add Google Test or Catch2 integration
    # find_package(GTest REQUIRED)
    # add_executable(test_inference tests/test_inference.cpp)
    # target_link_libraries(test_inference ${PROJECT_NAME}_lib GTest::GTest GTest::Main)
    # add_test(NAME test_inference COMMAND test_inference)
    message(STATUS "Tests enabled (stubs – add test sources in tests/)")
endif()

###############################################################################
# Ament package finalization
###############################################################################
ament_package()
