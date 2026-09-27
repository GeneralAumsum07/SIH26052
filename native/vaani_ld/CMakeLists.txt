# vaani_ld: the native low-delay runtime (low-delay plan Task 7).
#   native/vaani_ld/deps/fetch_ort.sh                         # pinned ONNX Runtime 1.30.0 into _deps/
#   cmake -S native/vaani_ld -B build/vaani_ld -DCMAKE_BUILD_TYPE=Release && cmake --build build/vaani_ld -j
# Options: VLD_ORT_DIR (the unpacked ONNX Runtime release), VLD_WITH_ALSA (live mode and the period test; ON by
# default and required on Linux targets), VLD_BUILD_TESTS. Cross builds: cmake/aarch64-linux-gnu.cmake.
cmake_minimum_required(VERSION 3.16)
project(vaani_ld CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O2 without fused multiply-add: the golden vectors pin float32 rounding (see src/dsp.cpp)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra -Wno-unused-parameter -ffp-contract=off)

set(VLD_ORT_VERSION 1.30.0)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ort_arch aarch64)
else()
  set(_ort_arch x64)
endif()
set(VLD_ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps/onnxruntime-linux-${_ort_arch}-${VLD_ORT_VERSION}"
    CACHE PATH "Unpacked ONNX Runtime ${VLD_ORT_VERSION} release")
option(VLD_WITH_ALSA "Build live mode and the ALSA period test" ON)
option(VLD_BUILD_TESTS "Build the golden and unit tests" ON)

if(NOT EXISTS "${VLD_ORT_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime ${VLD_ORT_VERSION} not found at ${VLD_ORT_DIR}.\n"
                      "Run native/vaani_ld/deps/fetch_ort.sh ${_ort_arch} or pass -DVLD_ORT_DIR=<release dir>.")
endif()
find_library(VLD_ORT_LIB onnxruntime PATHS "${VLD_ORT_DIR}/lib" NO_DEFAULT_PATH)
if(NOT VLD_ORT_LIB)
  message(FATAL_ERROR "libonnxruntime not found under ${VLD_ORT_DIR}/lib")
endif()
file(READ "${VLD_ORT_DIR}/VERSION_NUMBER" _ort_found_version)
string(STRIP "${_ort_found_version}" _ort_found_version)
if(NOT _ort_found_version STREQUAL VLD_ORT_VERSION)
  message(FATAL_ERROR "ONNX Runtime at ${VLD_ORT_DIR} is ${_ort_found_version}; ${VLD_ORT_VERSION} is pinned")
endif()

find_package(Threads REQUIRED)
if(VLD_WITH_ALSA)
  find_package(ALSA)
  if(NOT ALSA_FOUND)
    message(FATAL_ERROR "ALSA development files not found (libasound2-dev). Live mode needs them; "
                        "pass -DVLD_WITH_ALSA=OFF for a WAV-only build.")
  endif()
endif()

add_library(vaani_ld STATIC src/dsp.cpp src/sha256.cpp src/engine.cpp src/sim.cpp src/wav.cpp)
target_include_directories(vaani_ld PUBLIC include PRIVATE third_party/pocketfft)
target_include_directories(vaani_ld SYSTEM PUBLIC "${VLD_ORT_DIR}/include")
target_link_libraries(vaani_ld PUBLIC "${VLD_ORT_LIB}" Threads::Threads)
set_target_properties(vaani_ld PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_rpath "${VLD_ORT_DIR}/lib")
add_executable(vaani_ld_run src/main.cpp)
target_link_libraries(vaani_ld_run PRIVATE vaani_ld)
if(VLD_WITH_ALSA)
  target_sources(vaani_ld_run PRIVATE src/alsa_io.cpp)
  target_compile_definitions(vaani_ld_run PRIVATE VLD_WITH_ALSA=1)
  target_link_libraries(vaani_ld_run PRIVATE ALSA::ALSA)
  add_executable(vld_period_test bench/period_test.cpp src/alsa_io.cpp)
  target_link_libraries(vld_period_test PRIVATE vaani_ld ALSA::ALSA)
  set_target_properties(vld_period_test PROPERTIES BUILD_RPATH "${_rpath}")
endif()
set_target_properties(vaani_ld_run PROPERTIES BUILD_RPATH "${_rpath}")

add_executable(vld_step_bench bench/step_bench.cpp)
target_link_libraries(vld_step_bench PRIVATE vaani_ld)
set_target_properties(vld_step_bench PROPERTIES BUILD_RPATH "${_rpath}")
# the counting operator new/delete replacements trip a GCC false positive
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(vld_step_bench PRIVATE -Wno-mismatched-new-delete)
endif()

if(VLD_BUILD_TESTS)
  add_executable(vld_golden_test tests/golden_test.cpp)
  target_link_libraries(vld_golden_test PRIVATE vaani_ld)
  add_executable(vld_unit_test tests/unit_test.cpp)
  target_link_libraries(vld_unit_test PRIVATE vaani_ld)
  set_target_properties(vld_golden_test vld_unit_test PROPERTIES BUILD_RPATH "${_rpath}")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vld_unit_test PRIVATE -Wno-mismatched-new-delete)
  endif()
endif()

message(STATUS "vaani_ld: ONNX Runtime ${_ort_found_version} (${VLD_ORT_LIB}), ALSA ${VLD_WITH_ALSA}, "
               "${CMAKE_SYSTEM_PROCESSOR}, ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
