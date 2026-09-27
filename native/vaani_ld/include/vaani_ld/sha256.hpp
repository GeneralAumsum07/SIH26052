// SHA-256 (FIPS 180-4) for verifying shipped coefficient files and graphs on the board.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace vld {
std::string sha256_hex(const uint8_t* data, size_t n);
std::string sha256_file(const std::string& path);
}  // namespace vld
