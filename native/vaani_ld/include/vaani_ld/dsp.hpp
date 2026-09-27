// Low-delay DSP for the native runtime (low-delay plan Task 7): the contract, the asymmetric window pair, the r8
// limiter, the frontend, frame validity, analysis/synthesis and the 48 <-> 16 kHz polyphase resamplers.
// Every class ports its Python reference operation for operation (vaani/dsp/limiter_kernel.py, low_delay_frontend.py,
// low_delay_stft.py, vaani/live.py Decimate3/Interpolate3) and is checked against the Task 6 golden vectors.
// Buffers are sized at construction; no member allocates after that.
#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vld {

struct Contract {
    std::string id;
    int k = 512, hop = 0, support = 0;
    int limiter_sub = 32, ramp_samples = 3072;
    int crossfade() const { return support - hop; }
    int history() const { return k - hop; }
    int release_lead() const { return support - hop; }
    int bins() const { return k / 2 + 1; }
};

// "vaanife_ld_asym512_h96_s160_v1" -> K 512, H 96, L 160. Throws on a legacy or malformed id.
Contract parse_contract_id(const std::string& id);
// vaani.audio_contract._round_hash(a, s): SHA-256 of the windows rounded to 12 decimals (numpy's multiply-rint-divide).
std::string window_round_hash(const Contract& c);

// Readers for the flat JSON records the Python side writes (contract record, coefficient files). A missing key throws.
std::string json_string(const std::string& js, const std::string& key);
double json_number(const std::string& js, const std::string& key);
// (a, p, s) float64 windows, as vaani.audio_contract.ld_windows.
void ld_windows(const Contract& c, std::vector<double>& a, std::vector<double>& p, std::vector<double>& s);

// The r8 limiter (vaani.dsp.limiter defaults, fix_latch off) as the compiled kernel computes it: float32 pairwise
// sums in NumPy's order, float64 recurrence, gain applied as a float32 multiply.
struct LimiterState { bool has_env = false; double env = 0, floor = 0, gain = 1; int64_t engaged = 0, run = 0; };
class Limiter {
public:
    explicit Limiter(int sub = 32, int sr = 16000);
    void reset() { st = LimiterState{}; }
    void process(float* p, float* r, int n);   // in place
    LimiterState st;
private:
    int sub;
    double thr, rel, env_a, far, floor_up, floor_down, floor_max, min_env;
};

struct FrontendConfig { bool limiter = true; bool ref_policy = true; int ramp_samples = 3072; };

// LowDelayFrontend.process: finite fallback, reference zeroing, limiter, reconnect ramp (samples).
class Frontend {
public:
    Frontend(const Contract& c, const FrontendConfig& cfg);
    void reset();
    // prim/ref/avail: H samples in; out_p/out_r/valid: H out. Returns true on a primary discontinuity.
    bool process(const float* prim, const float* ref, const uint8_t* avail, float* out_p, float* out_r, uint8_t* valid);
    Limiter lim;
    int64_t since = 0, sample = 0, discontinuities = 0;
    bool prev_avail = true;
private:
    Contract c;
    FrontendConfig cfg;
};

// StreamValidity: frame j valid iff no unavailable real sample in its K-sample analysis support.
class Validity {
public:
    explicit Validity(const Contract& c) : c(c) { reset(); }
    void reset() { last_bad = -1000000000LL; sample = 0; }
    float push(const uint8_t* avail);
    int64_t last_bad, sample;
private:
    Contract c;
};

struct RealFft;   // a planned length-K real FFT (PocketFFT, float64), shared by the transforms of one contract

class Analyzer {
public:
    explicit Analyzer(const Contract& c);
    void reset();
    void push(const float* hop, std::complex<float>* spec);   // spec: bins() values
    std::vector<float> hist;
private:
    Contract c;
    std::vector<float> a, frame;
    std::vector<double> f64, scratch;
    std::shared_ptr<const RealFft> fft;
};

class Synthesizer {
public:
    explicit Synthesizer(const Contract& c);
    void reset();
    void push(const std::complex<float>* spec, float* out);    // out: H samples
    std::vector<float> pending;
private:
    Contract c;
    std::vector<float> s_tail, y;
    std::vector<double> y64, scratch;
    std::shared_ptr<const RealFft> fft;
};

// 48 kHz FIR resampler pair (vaani.resampler): coefficients from deploy/resampler/<id>.json, SHA-256 verified.
struct Fir { std::string id; std::vector<double> h; std::string sha256; double pair_peak_ms = 0; };
Fir load_fir(const std::string& json_path);
std::string coef_sha256(const std::vector<double>& h);

// Polyphase Decimate3: out[m] = sum_k h[k] x[3m - k] (x of the whole stream), float64 accumulation, float32 out.
class Decimate3 {
public:
    // max_block: the largest block process() accepts; its scratch buffer is sized here, so process() never allocates
    Decimate3(int channels, const std::vector<double>& h, int max_block = 4096);
    void reset();
    // in: channels x n (n % 3 == 0, n <= max_block), channel-major; out: channels x n/3
    void process(const float* in, int n, float* out);
    std::vector<double> state;   // channels x (taps - 1): the previous input samples
private:
    int ch, max_block; std::vector<double> h; std::vector<double> buf;
};

// Polyphase Interpolate3: zero-stuff by 3, filter with 3*h, float64 accumulation, float32 out.
class Interpolate3 {
public:
    Interpolate3(int channels, const std::vector<double>& h);
    void reset();
    void process(const float* in, int n, float* out);   // in: channels x n; out: channels x 3n
    std::vector<double> state;   // channels x history of 16 kHz input samples
private:
    int ch, hist; std::vector<double> h3;
};

}  // namespace vld
