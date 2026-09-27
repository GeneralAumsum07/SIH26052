# Cross toolchain for the Arm board (low-delay plan Task 7): Ubuntu/Debian g++-aarch64-linux-gnu.
#   native/vaani_ld/deps/fetch_ort.sh aarch64
#   cmake -S native/vaani_ld -B build/vaani_ld-arm64 -DCMAKE_TOOLCHAIN_FILE=native/vaani_ld/cmake/aarch64-linux-gnu.cmake \
#         -DCMAKE_BUILD_TYPE=Release [-DVLD_ALSA_ROOT=<arm64 sysroot with libasound>]
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu ${VLD_ALSA_ROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
