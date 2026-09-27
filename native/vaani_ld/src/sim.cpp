// See include/vaani_ld/sim.hpp.
#include "vaani_ld/sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace vld {

std::vector<std::string> check_timeline(const Timeline& t, int hop16, int support16, double pair_ms, Budget* b) {
    std::vector<std::string> why;
    const int N = 3 * hop16;
    if (t.period != 48 && t.period != 96)
        why.push_back("period " + std::to_string(t.period) + " frames: Section 4 allows 48 (1 ms) or 96 (2 ms)");
    if (t.period == 96 && support16 > 128)
        why.push_back("2 ms periods keep only L = 8 ms eligible (Section 4)");
    if (t.period <= 0 || N % t.period)
        why.push_back("the " + std::to_string(N) + "-frame hop is not a whole number of periods");
    else if (t.hop_periods != N / t.period)
        why.push_back("hop_periods " + std::to_string(t.hop_periods) + " != " + std::to_string(N / t.period));
    if (t.dproc_periods < 1 || t.dproc_periods * t.period > N)
        why.push_back("D_proc must be 1 .. hop whole periods (Section 4: it never exceeds H)");
    if (t.queue_periods < 2) why.push_back("the device must hold at least two periods (one playing, one queued)");
    if (b) {
        b->support_ms = support16 / 16.0;
        b->resampler_ms = pair_ms;
        b->io_ms = (t.dproc_periods + t.queue_periods - 1) * t.period / 48.0;
    }
    return why;
}

// ---- ring ----------------------------------------------------------------------------------------------------------
PeriodRing::PeriodRing(int capacity, int period) : cap(capacity), P(period) {
    if (cap < 1 || P < 1) throw std::runtime_error("PeriodRing needs a positive capacity and period");
    data.assign(static_cast<size_t>(cap) * P, 0.0f);
    idx.assign(cap, 0);
    ns.assign(cap, 0);
}

int PeriodRing::occupancy() const {
    return static_cast<int>(tail.load(std::memory_order_acquire) - head.load(std::memory_order_acquire));
}

bool PeriodRing::push(int64_t index, const float* d, uint64_t produced_ns) {
    const uint64_t t = tail.load(std::memory_order_relaxed);
    if (t - head.load(std::memory_order_acquire) >= static_cast<uint64_t>(cap)) return false;
    const size_t s = t % cap;
    std::memcpy(&data[s * P], d, sizeof(float) * P);
    idx[s] = index; ns[s] = produced_ns;
    tail.store(t + 1, std::memory_order_release);
    max_occupancy = std::max(max_occupancy, static_cast<int>(t + 1 - head.load(std::memory_order_relaxed)));
    return true;
}

PeriodRing::Take PeriodRing::take(int64_t index, float* out, int64_t* stale) {
    uint64_t h = head.load(std::memory_order_relaxed);
    while (h != tail.load(std::memory_order_acquire)) {
        const size_t s = h % cap;
        if (idx[s] < index) {                              // stale: its play time has passed
            ++h; head.store(h, std::memory_order_release);
            if (stale) ++*stale;
            continue;
        }
        if (idx[s] == index) {
            std::memcpy(out, &data[s * P], sizeof(float) * P);
            head.store(h + 1, std::memory_order_release);
            return TAKEN;
        }
        break;                                             // an early period: keep it, play silence now
    }
    std::memset(out, 0, sizeof(float) * P);
    return index < 0 ? SILENCE_START : LATE;
}

// ---- simulation --------------------------------------------------------------------------------------------------------
std::string SimResult::json() const {
    std::ostringstream o;
    o << "{\"hops\": " << hops << ", \"late_hops\": " << late_hops << ", \"stale_periods\": " << stale_periods
      << ", \"late_periods\": " << late_periods << ", \"silence_start_periods\": " << silence_start_periods
      << ", \"played_ok\": " << played_ok << ", \"misplaced\": " << misplaced << ", \"max_ring\": " << max_ring
      << ", \"max_device\": " << max_device << ", \"max_capture_backlog\": " << max_capture_backlog
      << ", \"recoveries\": " << recoveries << ", \"delay_frames\": " << delay_frames << "}";
    return o.str();
}

SimResult simulate(const Timeline& t, int64_t hops, const std::function<double(int64_t)>& proc_frames) {
    const int P = t.period, n = t.hop_periods, q = t.queue_periods;
    const int64_t N = t.hop_frames(), dp = t.delay_frames() / P;
    PeriodRing ring(t.ring_periods(), P);
    SimResult r;
    r.hops = hops;
    r.delay_frames = t.delay_frames();
    std::vector<float> buf(P), out(P);
    int64_t next_hop = 0;          // next hop to finish processing
    double finish_prev = 0;
    // processing is sequential on one thread: hop j starts when captured and when hop j-1 is done
    std::vector<double> finish(hops);
    for (int64_t j = 0; j < hops; ++j) {
        const double complete = static_cast<double>((j + 1) * N);
        const double start = std::max(complete, finish_prev);
        r.max_capture_backlog = std::max<int64_t>(r.max_capture_backlog, static_cast<int64_t>(std::ceil(start - complete)));
        finish[j] = start + std::max(0.0, proc_frames(j));
        finish_prev = finish[j];
    }
    const int64_t last_period = hops * n - 1;
    const int64_t k_end = last_period + dp - (q - 1);
    int64_t stale = 0;
    for (int64_t k = 0; k <= k_end; ++k) {
        const double now = static_cast<double>(k * P);
        while (next_hop < hops && finish[next_hop] <= now) {     // producer: hops finished by this boundary
            const int64_t j = next_hop++;
            if (finish[j] > static_cast<double>(t.deadline_frame(j))) {
                ++r.late_hops; ++r.recoveries;                    // stale work is dropped, the stream recovers
                continue;
            }
            for (int i = 0; i < n; ++i) {
                const int64_t m = j * n + i;
                std::fill(buf.begin(), buf.end(), static_cast<float>(m));   // tag: the period's output index
                if (!ring.push(m, buf.data(), static_cast<uint64_t>(finish[j]))) ++r.misplaced;
            }
            r.max_ring = std::max<int64_t>(r.max_ring, ring.occupancy());
        }
        // period writer: at boundary 0 it prefills q periods, then writes one period per boundary
        const int64_t w_lo = k == 0 ? 0 : k + q - 1, w_hi = k + q - 1;
        for (int64_t w = w_lo; w <= w_hi; ++w) {
            const int64_t m = w - dp;
            switch (ring.take(m, out.data(), &stale)) {
                case PeriodRing::TAKEN:
                    if (out[0] == static_cast<float>(m) && out[P - 1] == static_cast<float>(m)) ++r.played_ok;
                    else ++r.misplaced;
                    break;
                case PeriodRing::SILENCE_START: ++r.silence_start_periods; break;
                case PeriodRing::LATE: if (m <= last_period) ++r.late_periods; break;
            }
        }
        r.max_device = std::max<int64_t>(r.max_device, q);
    }
    for (; next_hop < hops; ++next_hop) { ++r.late_hops; ++r.recoveries; }   // never delivered before the end
    r.stale_periods = stale;
    return r;
}

}  // namespace vld
