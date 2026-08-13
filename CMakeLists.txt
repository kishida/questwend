cmake_minimum_required(VERSION 3.20)
project(questwend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- options ----
option(QW_CUDA  "Enable CUDA backend"  OFF)
option(QW_METAL "Enable Metal backend" OFF)

if(QW_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # ggml defaults CUDA graphs OFF for standalone builds ("llama.cpp only");
    # enable them to eliminate per-kernel launch overhead during decode.
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
if(QW_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# ---- vendored ggml (self-contained, see third_party/ggml/GGML_VERSION) ----
add_subdirectory(third_party/ggml)

# ---- core engine library ----
add_library(questwend_core STATIC
    core/gguf_util.cpp
    core/model.cpp
    core/tokenizer.cpp
    core/runtime.cpp
    core/expert_cache.cpp
    core/sampler.cpp
    core/chat.cpp
    core/vision.cpp
)
target_include_directories(questwend_core PUBLIC core)
target_include_directories(questwend_core PRIVATE third_party/stb)
target_link_libraries(questwend_core PUBLIC ggml)

# ---- CLI ----
add_executable(qw-cli cli/main.cpp)
target_link_libraries(qw-cli PRIVATE questwend_core)

# ---- tokenizer cross-check tool (tests/tokenize_main.cpp) ----
add_executable(qw-tokenize tests/tokenize_main.cpp)
target_link_libraries(qw-tokenize PRIVATE questwend_core)

# ---- Server (OpenAI-compatible) ----
add_executable(qw-server server/main.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-server PRIVATE
    third_party/cpp-httplib
    third_party/nlohmann
)
target_link_libraries(qw-server PRIVATE questwend_core)
find_package(Threads REQUIRED)
target_link_libraries(qw-server PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(qw-server PRIVATE ws2_32)
endif()

# ---- keep the ggml DLLs next to the executables (Windows) ----
# ggml builds its DLLs into build/bin/<config> while the executables land in
# build/<config>, so Windows resolves them from the exe directory instead --
# a stale copy there silently shadows a freshly built ggml. Copy them after
# every link so the two can never drift apart.
if(WIN32)
    foreach(tgt qw-cli qw-server qw-tokenize)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${tgt}> $<TARGET_FILE_DIR:${tgt}>
            COMMAND_EXPAND_LISTS)
    endforeach()
endif()
