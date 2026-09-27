// The release timeline (Task 7 / Section 4) and the pieces the live loop shares with its deterministic simulation:
// a bounded, timestamped single-producer single-consumer period ring and the period writer's decision.
//
// Timeline, in 48 kHz frames from the linked start (capture and playback start on the same frame):
//   P = period frames, n = hop periods (3H = nP), d = D_proc in whole periods (dP <= 3H), q = playback periods the
//   writer keeps queued in the device (2: one playing, one waiting).
//   Capture hop j is complete at (j+1)nP; its processing must end by (j+1)nP + dP (its deadline).
//   Output frame m (the released stream, 48 kHz) is written to the device at m + delay - (q-1)P and plays at
//   m + delay, with delay = nP + (d + q - 1)P. The first `delay` frames played are silence (the startup threshold).
//   End to end, the path delay is 3L + (d + q - 1)P frames plus the resampler pair and the converters: with q = 2
//   that is L + D_proc + one playback period (L_io = D_proc + P).
//   Occupancy: the device holds q periods; the ring holds at most n + d periods (a hop's n plus the d periods of the
//   previous hop not yet written), so its capacity bounds, and never adds, latency.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vld {

struct Timeline {
    int period = 48;          // P, frames at 48 kHz
    int hop_periods = 6;      // n
    int dproc_periods = 1;    // d
    int queue_periods = 2;    // q
    int hop_frames() const { return period * hop_periods; }
    int64_t delay_frames() const { return static_cast<int64_t>(period) * (hop_periods + dproc_periods + queue_periods - 1); }
    int ring_periods() const { return hop_periods + dproc_periods + 1; }   // capacity: the bound plus one
    int64_t deadline_frame(int64_t hop) const { return (hop + 1) * hop_frames() + static_cast<int64_t>(dproc_periods) * period; }
};

// Section 4 budget of a live setting (ms). Upper estimate = L + resampler pair + (d + q - 1)P + converters + FIFO.
struct Budget {
    double support_ms = 0, resampler_ms = 0, io_ms = 0, converters_ms = 0.5, fifo_ms = 0.1;
    double total_ms() const { return support_ms + resampler_ms + io_ms + converters_ms + fifo_ms; }
    bool eligible() const { return total_ms() <= 13.0 + 1e-9; }   // Section 4 eligibility rule
};

// Checks a live setting against Section 4: whole periods per hop, D_proc within a hop, 1 ms periods (2 ms only as
// the Section 4 fallback, which keeps L = 8 ms alone). Returns the reasons it cannot run (empty = runs), and fills
// the budget; eligibility (<= 13.0 ms) is reported separately because R0, the control, is never eligible.
std::vector<std::string> check_timeline(const Timeline& t, int hop16, int support16, double resampler_pair_ms,
                                        Budget* budget);

// Bounded SPSC ring of output periods, each stamped with its output period index and the time it was produced.
class PeriodRing {
public:
    PeriodRing(int capacity, int period);
    bool push(int64_t index, const float* data, uint64_t produced_ns);   // false when full (never blocks)
    // the period writer: fill `out` (P frames) for output period `index`; returns what happened
    enum Take { TAKEN, SILENCE_START, LATE };
    Take take(int64_t index, float* out, int64_t* stale_dropped);
    int occupancy() const;
    int capacity() const { return cap; }
    int max_occupancy = 0;
private:
    int cap, P;
    std::vector<float> data;
    std::vector<int64_t> idx;
    std::vector<uint64_t> ns;
    std::atomic<uint64_t> head{0}, tail{0};   // tail: next write (producer), head: next read (consumer)
};

// Deterministic simulated device: capture complete every hop, processing times from `proc_frames(hop)` (frames,
// may exceed a hop), the period writer at every period boundary. Late hops are dropped by the producer (as the live
// loop does) and counted; each late hop starts a recovery.
struct SimResult {
    int64_t hops = 0, late_hops = 0, stale_periods = 0, late_periods = 0, silence_start_periods = 0;
    int64_t played_ok = 0, misplaced = 0, max_ring = 0, max_device = 0, max_capture_backlog = 0, recoveries = 0;
    int64_t delay_frames = 0;
    std::string json() const;
};
SimResult simulate(const Timeline& t, int64_t hops, const std::function<double(int64_t)>& proc_frames);

}  // namespace vld
