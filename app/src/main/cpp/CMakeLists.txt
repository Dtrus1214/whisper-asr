cmake_minimum_required(VERSION 3.22.1)
project(crystal_asr)

# ------------------------------------------------------------------------------
# Notes on linking external libraries
# ------------------------------------------------------------------------------
# You can link either:
#   1) a source-tree dependency (built by CMake) via `add_subdirectory(...)`
#   2) a prebuilt external library (static `.a` or shared `.so`) via IMPORTED targets
#
# 1) Source-tree dependency (recommended)
#    - Vendor/copy the dependency into your repo (e.g. `third_party/somelib/`)
#    - Ensure it has its own `CMakeLists.txt`
#    - Then do:
#         add_subdirectory(third_party/somelib)
#         target_link_libraries(asr somelib_target)
#
# 2) Prebuilt static library (a `.a` file)
#    - Place the `.a` files somewhere in your project (example: `third_party/mylib/lib/${ANDROID_ABI}/libmylib.a`)
#    - Declare an IMPORTED target and set its location per ABI:
#
#         add_library(mylib STATIC IMPORTED)
#         set_target_properties(mylib PROPERTIES
#             IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mylib/lib/${ANDROID_ABI}/libmylib.a"
#         )
#         target_link_libraries(asr mylib)
#
# 3) Prebuilt shared library (a `.so` file)
#    - Place `.so` files in `src/main/jniLibs/<abi>/libmylib.so`, OR reference a custom path similar to the static case above.
#    - Declare an IMPORTED target and set its location per ABI:
#
#         add_library(mylib SHARED IMPORTED)
#         set_target_properties(mylib PROPERTIES
#             IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mylib/lib/${ANDROID_ABI}/libmylib.so"
#         )
#         target_link_libraries(asr mylib)
#
# 4) Include directories for headers
#    - If an external library provides headers:
#         target_include_directories(asr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mylib/include")
#
# 5) If your prebuilt library needs system libs
#    - You can link those too (examples):
#         target_link_libraries(asr android log)
#         target_link_libraries(asr ${log-lib})
#
# ------------------------------------------------------------------------------

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
