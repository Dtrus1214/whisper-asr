cmake_minimum_required(VERSION 3.22.1)
project(crystal_asr)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp)

add_library(asr SHARED asr_jni.cpp)
target_link_libraries(asr whisper)

find_library(log-lib log)
target_link_libraries(asr ${log-lib})

target_compile_features(asr PRIVATE cxx_std_17)
