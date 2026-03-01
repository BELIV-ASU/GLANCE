cmake_minimum_required(VERSION 3.20)
project(tinyvlm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────
option(GGML_CUDA       "Enable CUDA backend in llama.cpp"  ON)
option(BUILD_SHARED_LIBS "Build shared libraries"          OFF)

# ─── Fetch llama.cpp (includes multimodal / mtmd) ────────
include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# llama.cpp build options
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

if(GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

set(LLAMA_MMAP  ON CACHE BOOL "" FORCE)
set(LLAMA_MLOCK ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# ─── tinyvlm executable ─────────────────────────────────
add_executable(tinyvlm
    src/main.cpp
    src/vlm.cpp
)

target_include_directories(tinyvlm PRIVATE
    src/
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/common
    ${llama_cpp_SOURCE_DIR}/tools/mtmd
)

target_link_libraries(tinyvlm PRIVATE
    llama
    ggml
    common
    mtmd
)

if(MSVC)
    target_compile_options(tinyvlm PRIVATE /W3 /O2)
else()
    target_compile_options(tinyvlm PRIVATE -Wall -Wextra -O2)
endif()

# ─── Install ─────────────────────────────────────────────
install(TARGETS tinyvlm RUNTIME DESTINATION bin)
