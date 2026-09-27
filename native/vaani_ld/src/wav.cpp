#include "vaani_ld/wav.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vld {

Wav read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char id[4];
    uint32_t sz;
    f.read(id, 4); f.read(reinterpret_cast<char*>(&sz), 4);
    if (std::memcmp(id, "RIFF", 4) != 0) throw std::runtime_error(path + ": not a RIFF file");
    f.read(id, 4);
    if (std::memcmp(id, "WAVE", 4) != 0) throw std::runtime_error(path + ": not a WAVE file");
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    Wav w;
    while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&sz), 4)) {
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<char> b(sz);
            f.read(b.data(), sz);
            std::memcpy(&fmt, &b[0], 2); std::memcpy(&ch, &b[2], 2); std::memcpy(&sr, &b[4], 4); std::memcpy(&bits, &b[14], 2);
            if (fmt == 0xFFFE && sz >= 26) std::memcpy(&fmt, &b[24], 2);   // WAVE_FORMAT_EXTENSIBLE sub-format
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!ch) throw std::runtime_error(path + ": data before fmt");
            const bool pcm16 = fmt == 1 && bits == 16, f32 = fmt == 3 && bits == 32;
            if (!pcm16 && !f32) throw std::runtime_error(path + ": only 16-bit PCM and 32-bit float WAV are read");
            const size_t n = sz / (bits / 8) / ch;
            std::vector<char> b(sz);
            f.read(b.data(), sz);
            w.sr = static_cast<int>(sr); w.channels = ch;
            w.ch.assign(ch, std::vector<float>(n));
            for (size_t i = 0; i < n; ++i)
                for (int c = 0; c < ch; ++c) {
                    if (pcm16) { int16_t v; std::memcpy(&v, &b[(i * ch + c) * 2], 2); w.ch[c][i] = v / 32768.0f; }
                    else std::memcpy(&w.ch[c][i], &b[(i * ch + c) * 4], 4);
                }
            return w;
        } else {
            f.seekg(sz + (sz & 1), std::ios::cur);
        }
    }
    throw std::runtime_error(path + ": no data chunk");
}

void write_wav(const std::string& path, const Wav& w) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint16_t ch = static_cast<uint16_t>(w.channels), fmt = 3, bits = 32, align = ch * 4;
    const uint32_t sr = w.sr, rate = sr * align, data = static_cast<uint32_t>(w.frames() * align), riff = 36 + data, fsz = 16;
    f.write("RIFF", 4); f.write(reinterpret_cast<const char*>(&riff), 4); f.write("WAVE", 4);
    f.write("fmt ", 4); f.write(reinterpret_cast<const char*>(&fsz), 4);
    f.write(reinterpret_cast<const char*>(&fmt), 2); f.write(reinterpret_cast<const char*>(&ch), 2);
    f.write(reinterpret_cast<const char*>(&sr), 4); f.write(reinterpret_cast<const char*>(&rate), 4);
    f.write(reinterpret_cast<const char*>(&align), 2); f.write(reinterpret_cast<const char*>(&bits), 2);
    f.write("data", 4); f.write(reinterpret_cast<const char*>(&data), 4);
    for (size_t i = 0; i < w.frames(); ++i)
        for (int c = 0; c < w.channels; ++c) f.write(reinterpret_cast<const char*>(&w.ch[c][i]), 4);
}

}  // namespace vld
