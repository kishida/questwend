cmake_minimum_required(VERSION 3.20)
project(qwencpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- options ----
option(QWENCPP_CUDA  "Enable CUDA backend"  OFF)
option(QWENCPP_METAL "Enable Metal backend" OFF)

if(QWENCPP_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # ggml defaults CUDA graphs OFF for standalone builds ("llama.cpp only");
    # enable them to eliminate per-kernel launch overhead during decode.
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
if(QWENCPP_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# ---- vendored ggml (self-contained, see third_party/ggml/GGML_VERSION) ----
add_subdirectory(third_party/ggml)

# ---- core engine library ----
add_library(qwencpp_core STATIC
    core/gguf_util.cpp
    core/model.cpp
    core/tokenizer.cpp
    core/runtime.cpp
    core/expert_cache.cpp
    core/sampler.cpp
    core/chat.cpp
)
target_include_directories(qwencpp_core PUBLIC core)
target_link_libraries(qwencpp_core PUBLIC ggml)

# ---- CLI ----
add_executable(infer cli/main.cpp)
target_link_libraries(infer PRIVATE qwencpp_core)

# ---- Server (OpenAI-compatible) ----
add_executable(infer-server server/main.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(infer-server PRIVATE
    third_party/cpp-httplib
    third_party/nlohmann
)
target_link_libraries(infer-server PRIVATE qwencpp_core)
find_package(Threads REQUIRED)
target_link_libraries(infer-server PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(infer-server PRIVATE ws2_32)
endif()
