// RIFF WAVE files for WAV mode: 16-bit PCM or 32-bit float in, 32-bit float out. Channel-major in memory.
#pragma once
#include <string>
#include <vector>

namespace vld {

struct Wav {
    int sr = 16000, channels = 1;
    std::vector<std::vector<float>> ch;   // channels x frames
    size_t frames() const { return ch.empty() ? 0 : ch[0].size(); }
};

Wav read_wav(const std::string& path);
void write_wav(const std::string& path, const Wav& w);   // IEEE float32

}  // namespace vld
