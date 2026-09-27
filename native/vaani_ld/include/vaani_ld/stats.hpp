// Allocation-free latency histogram (1 us bins up to 50 ms, then an overflow bin) for the audio hot path, with
// exact max and mean; percentiles are read from the bins (upper bin edge, so they never understate).
#pragma once
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace vld {

class LatencyHist {
public:
    static constexpr int BINS = 50000;   // 1 us each
    LatencyHist() : bins(BINS + 1, 0) {}
    void add(uint64_t ns) {
        const uint64_t us = ns / 1000;
        ++bins[us < BINS ? us : BINS];
        ++n; sum += ns; mx = std::max(mx, ns);
    }
    uint64_t count() const { return n; }
    double max_ms() const { return mx / 1e6; }
    double mean_ms() const { return n ? static_cast<double>(sum) / n / 1e6 : 0.0; }
    double pct_ms(double p) const {
        if (!n) return 0.0;
        const uint64_t want = static_cast<uint64_t>(p / 100.0 * n + 0.5);
        uint64_t acc = 0;
        for (int i = 0; i <= BINS; ++i) {
            acc += bins[i];
            if (acc >= std::max<uint64_t>(want, 1)) return i < BINS ? (i + 1) / 1000.0 : max_ms();
        }
        return max_ms();
    }
    uint64_t over(uint64_t ns) const {   // samples strictly above ns (bin resolution, exact max)
        uint64_t c = 0;
        for (int i = static_cast<int>(std::min<uint64_t>(ns / 1000, BINS)); i <= BINS; ++i) c += bins[i];
        return mx > ns ? c : 0;
    }
    std::string json() const {
        std::ostringstream o;
        o << "{\"n\": " << n << ", \"max_ms\": " << max_ms() << ", \"p999_ms\": " << pct_ms(99.9)
          << ", \"p99_ms\": " << pct_ms(99.0) << ", \"mean_ms\": " << mean_ms() << "}";
        return o.str();
    }
private:
    std::vector<uint32_t> bins;
    uint64_t n = 0, sum = 0, mx = 0;
};

}  // namespace vld
