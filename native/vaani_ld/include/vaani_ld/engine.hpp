// One low-delay stream (native twin of vaani.low_delay_live.LowDelayStreamEngine, guards off):
// frontend -> validity -> analysis -> ONNX Runtime step -> synthesis, one H-sample hop at a time, plus an optional
// 48 kHz runner (Decimate3 before, Interpolate3 after). Every buffer and ORT tensor is created in the constructor;
// process_hop allocates nothing on the host side (the inference library's own behaviour is measured separately by
// bench/step_bench). The caller's inference thread must call flush_denormals() once (FPCR.FZ / FTZ+DAZ).
//
// Recovery (Task 7, native only): begin_recovery() drops the stream state and moves through a delay-matched primary
// bypass. The released output is the frontend's primary delayed by L - H samples (the model path's own alignment)
// for ceil((K - H) / H) hops while the model refills its analysis history, then one hop crosses linearly into the
// model output. A hop whose primary is invalid (non-finite) releases silence. Every recovery is counted.
#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vaani_ld/dsp.hpp"

namespace vld {

// Set flush-to-zero (and denormals-are-zero on x86) on the calling thread; returns the previous control word.
uint64_t flush_denormals(bool on = true);
void restore_fp_control(uint64_t word);
uint64_t now_ns();   // CLOCK_MONOTONIC

struct EngineConfig {
    FrontendConfig frontend;
    int threads = 1;
    bool denormal_as_zero = true;   // ORT session option (x86 only; aarch64 relies on FPCR.FZ on this thread)
    bool timing = false;            // per-stage clock reads into Engine::stage_ns
};

// Per-hop stage outputs for golden-vector comparison (filled when a Trace is attached).
struct Trace {
    std::vector<float> frontend_out;              // 2 x H
    float frame_valid = 0;
    std::vector<std::complex<float>> p, r;        // analysis spectra
    std::vector<float> step_out;                  // 257 x 2 (re, im)
    bool discontinuity = false;
};

// Last hop's processing time per stage (ns), when EngineConfig::timing is set.
struct StageNs { uint64_t frontend = 0, analysis = 0, step = 0, synthesis = 0, resample = 0, total = 0; };

class OrtStep;   // ORT session and its preallocated tensors (engine.cpp)

class Engine {
public:
    // onnx: a contract-stamped VaaniFE step graph; expected_contract: the id the caller runs (sidecar); the graph's
    // audio_contract_id metadata must equal it (an unstamped graph is refused), its record must hash to the stamped
    // audio_contract_hash and describe this contract, windows included.
    Engine(const std::string& onnx, const std::string& expected_contract, const EngineConfig& cfg = {});
    ~Engine();
    const Contract& contract() const { return c; }
    int state_floats() const;
    std::string graph_contract_hash() const;

    void reset();
    // one hop: prim/ref H samples, avail H flags -> out H samples
    void process_hop(const float* prim, const float* ref, const uint8_t* avail, float* out);
    // end of stream: one hop of known zeros (frontend bypassed) -> out H samples
    void flush_hop(float* out);
    // drop the stream state and release the delay-matched primary until the model has refilled its history
    void begin_recovery();
    bool in_recovery() const { return bypass_left > 0 || xfade_pending; }
    int bypass_hops() const;

    // whole-stream state (frontend, limiter, validity, analysis, synthesis, neural state, bypass, counters). The
    // blob names its contract, window hash and configuration; load_state refuses any other.
    std::vector<uint8_t> save_state() const;
    void load_state(const std::vector<uint8_t>& blob);

    Trace* trace = nullptr;
    int64_t hops = 0, discontinuity_hops = 0, recoveries = 0;
    StageNs stage_ns;
    std::vector<float>& neural_state();

private:
    std::string config_tag() const;
    Contract c;
    EngineConfig cfg;
    Frontend fe;
    Validity val;
    Analyzer an_p, an_r;
    Synthesizer syn;
    std::unique_ptr<OrtStep> ort;
    std::vector<float> fp, fr;           // frontend output
    std::vector<uint8_t> fv;
    std::vector<std::complex<float>> P, R, Y;
    // bypass: primary delay line (L - H samples), hops left, crossfade flag, scratch
    std::vector<float> bline, bout;
    int bypass_left = 0;
    bool xfade_pending = false;
};

// 48 kHz runner: 3H samples per channel in and out, through a resampler pair.
class Engine48 {
public:
    Engine48(Engine& e, const Fir& fir);
    void reset();
    void process(const float* prim48, const float* ref48, const uint8_t* avail48, float* out48);
    void flush(float* out48);
    std::vector<uint8_t> save_state() const;          // the engine's blob plus the resampler states, by FIR hash
    void load_state(const std::vector<uint8_t>& blob);
    Engine& e;
    Fir fir;
    Decimate3 dec;
    Interpolate3 itp;
private:
    std::vector<float> in48, x16, y16;
    std::vector<uint8_t> av16;
};

}  // namespace vld
