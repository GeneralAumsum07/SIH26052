// Minimal .npy reader for the golden-vector tests (little-endian, C order; float32, float64, int64, bool).
// tests/test_low_delay_native.py extracts each golden .npz into a directory of .npy files.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
    std::string dtype;            // "<f4", "<f8", "<i8", "|b1"
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t size() const { size_t n = 1; for (size_t s : shape) n *= s; return n; }
    std::vector<float> f32() const {
        std::vector<float> v(size());
        if (dtype == "<f4") std::memcpy(v.data(), data.data(), v.size() * 4);
        else if (dtype == "<f8") { const double* d = reinterpret_cast<const double*>(data.data()); for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(d[i]); }
        else if (dtype == "|b1") { for (size_t i = 0; i < v.size(); ++i) v[i] = data[i] ? 1.0f : 0.0f; }
        else throw std::runtime_error("npy: cannot read " + dtype + " as float32");
        return v;
    }
    std::vector<double> f64() const {
        std::vector<double> v(size());
        if (dtype == "<f8") std::memcpy(v.data(), data.data(), v.size() * 8);
        else { auto f = f32(); for (size_t i = 0; i < v.size(); ++i) v[i] = f[i]; }
        return v;
    }
    std::vector<int64_t> i64() const {
        std::vector<int64_t> v(size());
        if (dtype == "<i8") std::memcpy(v.data(), data.data(), v.size() * 8);
        else if (dtype == "|b1") for (size_t i = 0; i < v.size(); ++i) v[i] = data[i];
        else throw std::runtime_error("npy: cannot read " + dtype + " as int64");
        return v;
    }
    std::vector<uint8_t> u8() const {
        if (dtype != "|b1") throw std::runtime_error("npy: not a bool array");
        return data;
    }
};

inline Array load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic in " + path);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hlen = h; }
    else f.read(reinterpret_cast<char*>(&hlen), 4);
    std::string hdr(hlen, '\0');
    f.read(&hdr[0], hlen);
    Array a;
    auto val = [&](const std::string& key) {
        size_t p = hdr.find("'" + key + "'");
        if (p == std::string::npos) throw std::runtime_error("npy: header lacks " + key);
        return hdr.substr(hdr.find(':', p) + 1);
    };
    std::string d = val("descr");
    a.dtype = d.substr(d.find('\'') + 1, d.find('\'', d.find('\'') + 1) - d.find('\'') - 1);
    if (val("fortran_order").find("False") != 0 && val("fortran_order").find(" False") != 0)
        throw std::runtime_error("npy: Fortran order unsupported");
    std::string s = val("shape");
    s = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_not_of(' ') != std::string::npos) a.shape.push_back(std::stoul(tok));
    const size_t item = a.dtype == "|b1" ? 1 : a.dtype.back() - '0';
    a.data.resize(a.size() * item);
    f.read(reinterpret_cast<char*>(a.data.data()), a.data.size());
    if (!f) throw std::runtime_error("npy: truncated " + path);
    return a;
}

}  // namespace npy
