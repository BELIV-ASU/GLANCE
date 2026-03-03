cmake_minimum_required(VERSION 3.20)
project(tinyvlm_jetson LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Jetson AGX Orin = sm_87 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "Jetson AGX Orin GPU arch")

# ─── Options ──────────────────────────────────────────────
option(GGML_CUDA       "Enable CUDA backend (Jetson AGX Orin)"  ON)
option(BUILD_SHARED_LIBS "Build shared libraries"                OFF)

# ─── Find system packages ────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(CUDAToolkit REQUIRED)

# ─── Fetch llama.cpp ─────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)

if(GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

set(LLAMA_MMAP  ON CACHE BOOL "" FORCE)
set(LLAMA_MLOCK ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# ─── NVIDIA NPP libraries (for CUDA image preprocessing) ─
set(NPP_LIBS
    CUDA::nppc        # NPP core
    CUDA::nppicc      # NPP color conversion
    CUDA::nppig       # NPP geometry transforms (resize)
    CUDA::nppidei     # NPP data exchange & initialization
)

# ─── VLM library (shared between targets) ────────────────
add_library(vlm_jetson STATIC
    src/vlm.cpp
)
target_include_directories(vlm_jetson PUBLIC
    src/
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/common
    ${llama_cpp_SOURCE_DIR}/tools/mtmd
)
target_link_libraries(vlm_jetson PUBLIC
    llama
    ggml
    common
    mtmd
)
target_compile_options(vlm_jetson PRIVATE -Wall -Wextra -O2)

# ─── CUDA preprocessing library ─────────────────────────
add_library(cuda_preprocess STATIC
    src/cuda_preprocess.cu
)
target_include_directories(cuda_preprocess PUBLIC
    src/
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(cuda_preprocess PUBLIC
    CUDA::cudart
    ${NPP_LIBS}
)
set_target_properties(cuda_preprocess PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ─── ROS2 inference node ────────────────────────────────
add_executable(ros2_inference_node
    src/ros2_node.cpp
)
target_include_directories(ros2_inference_node PRIVATE
    src/
)
target_link_libraries(ros2_inference_node
    vlm_jetson
    cuda_preprocess
    ${OpenCV_LIBS}
)
ament_target_dependencies(ros2_inference_node
    rclcpp
    sensor_msgs
    cv_bridge
    image_transport
)
target_compile_options(ros2_inference_node PRIVATE -Wall -Wextra -O2)

# ─── Install ─────────────────────────────────────────────
install(TARGETS ros2_inference_node
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
