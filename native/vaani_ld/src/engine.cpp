// See include/vaani_ld/engine.hpp.
#include "vaani_ld/engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <stdexcept>

#include "vaani_ld/sha256.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace vld {

// ---- floating-point control ----------------------------------------------------------------------------------
uint64_t flush_denormals(bool on) {
#if defined(__aarch64__)
    uint64_t w;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(w));
    const uint64_t n = on ? (w | (1ULL << 24)) : (w & ~(1ULL << 24));   // FPCR.FZ
    __asm__ __volatile__("msr fpcr, %0" : : "r"(n));
    return w;
#elif defined(__x86_64__) || defined(__i386__)
    const unsigned w = _mm_getcsr();
    const unsigned bits = (1u << 15) | (1u << 6);                         // FTZ | DAZ
    _mm_setcsr(on ? (w | bits) : (w & ~bits));
    return w;
#else
    (void)on;
    return 0;
#endif
}

void restore_fp_control(uint64_t w) {
#if defined(__aarch64__)
    __asm__ __volatile__("msr fpcr, %0" : : "r"(w));
#elif defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(static_cast<unsigned>(w));
#else
    (void)w;
#endif
}

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ---- ONNX Runtime step ---------------------------------------------------------------------------------------
static Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "vaani_ld");
    return env;
}

class OrtStep {
public:
    OrtStep(const std::string& path, const EngineConfig& cfg) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(cfg.threads);
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cfg.denormal_as_zero) so.AddConfigEntry("session.set_denormal_as_zero", "1");
        so.AddConfigEntry("session.intra_op.allow_spinning", "0");
        sess = std::make_unique<Ort::Session>(ort_env(), path.c_str(), so);

        Ort::AllocatorWithDefaultOptions alloc;
        std::vector<std::string> ins, outs;
        for (size_t i = 0; i < sess->GetInputCount(); ++i) ins.push_back(sess->GetInputNameAllocated(i, alloc).get());
        for (size_t i = 0; i < sess->GetOutputCount(); ++i) outs.push_back(sess->GetOutputNameAllocated(i, alloc).get());
        const bool with_valid = ins == std::vector<std::string>{"spec", "valid", "state"};
        if (!(with_valid || ins == std::vector<std::string>{"spec", "state"}) ||
            outs != std::vector<std::string>{"spec_out", "state_out"})
            throw std::runtime_error(path + ": not a VaaniFE step graph");
        takes_valid = with_valid;
        const auto spec_shape = sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto st_shape = sess->GetInputTypeInfo(ins.size() - 1).GetTensorTypeAndShapeInfo().GetShape();
        if (spec_shape != std::vector<int64_t>{1, 4, 257})
            throw std::runtime_error(path + ": spec input is not (1, 4, 257); the low-delay route runs inputs 'pr'");
        if (st_shape.size() != 2 || st_shape[0] != 1 || st_shape[1] <= 0)
            throw std::runtime_error(path + ": state input is not a static (1, S) tensor");
        S = st_shape[1];

        Ort::ModelMetadata md = sess->GetModelMetadata();
        for (const char* k : {"audio_contract_id", "audio_contract_hash", "audio_contract"}) {
            auto v = md.LookupCustomMetadataMapAllocated(k, alloc);
            meta.push_back(v ? std::string(v.get()) : std::string());
        }

        spec.assign(4 * 257, 0.0f); out.assign(2 * 257, 0.0f); valid[0] = 1.0f;
        st[0].assign(S, 0.0f); st[1].assign(S, 0.0f);
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t sh_spec[3] = {1, 4, 257}, sh_valid[2] = {1, 1}, sh_st[2] = {1, S}, sh_out[3] = {1, 2, 257};
        v_spec = Ort::Value::CreateTensor<float>(mem, spec.data(), spec.size(), sh_spec, 3);
        v_valid = Ort::Value::CreateTensor<float>(mem, valid, 1, sh_valid, 2);
        v_out = Ort::Value::CreateTensor<float>(mem, out.data(), out.size(), sh_out, 3);
        for (int i = 0; i < 2; ++i) v_st[i] = Ort::Value::CreateTensor<float>(mem, st[i].data(), S, sh_st, 2);
        // ping-pong: binding i reads state i and writes state 1 - i; no tensor is created per hop
        for (int i = 0; i < 2; ++i) {
            bind[i] = std::make_unique<Ort::IoBinding>(*sess);
            bind[i]->BindInput("spec", v_spec);
            if (takes_valid) bind[i]->BindInput("valid", v_valid);
            bind[i]->BindInput("state", v_st[i]);
            bind[i]->BindOutput("spec_out", v_out);
            bind[i]->BindOutput("state_out", v_st[1 - i]);
        }
    }

    // spec (4 x 257, channel-major) and valid are written by the caller; out (2 x 257) is read after
    void run() {
        sess->Run(ropt, *bind[cur]);
        cur = 1 - cur;
    }
    void reset() {
        std::fill(st[0].begin(), st[0].end(), 0.0f);
        std::fill(st[1].begin(), st[1].end(), 0.0f);
        cur = 0;
    }
    std::vector<float>& state() { return st[cur]; }

    std::unique_ptr<Ort::Session> sess;
    Ort::RunOptions ropt;
    std::vector<std::string> meta;   // id, hash, record
    bool takes_valid = false;
    int64_t S = 0;
    std::vector<float> spec, out, st[2];
    float valid[1];
    int cur = 0;
    Ort::Value v_spec{nullptr}, v_valid{nullptr}, v_out{nullptr}, v_st[2] = {Ort::Value{nullptr}, Ort::Value{nullptr}};
    std::unique_ptr<Ort::IoBinding> bind[2];
};

// ---- engine -----------------------------------------------------------------------------------------------------
static void check_record(const std::string& where, const std::vector<std::string>& meta, const Contract& c,
                         const EngineConfig& cfg) {
    const std::string &id = meta[0], &hash = meta[1], &rec = meta[2];
    if (id.empty())
        throw std::runtime_error(where + ": no audio_contract_id metadata; an unstamped graph is never accepted as "
                                 "the low-delay contract " + c.id);
    if (hash.empty() || rec.empty())
        throw std::runtime_error(where + ": audio_contract_id is stamped without its record and hash");
    if (id != c.id)
        throw std::runtime_error(where + ": stamped for " + id + " but " + c.id + " is expected");
    const std::string h = sha256_hex(reinterpret_cast<const uint8_t*>(rec.data()), rec.size()).substr(0, 16);
    if (h != hash) throw std::runtime_error(where + ": contract record does not hash to the stamped hash");
    auto num = [&](const char* k) { return static_cast<int>(json_number(rec, k)); };
    if (json_string(rec, "audio_contract_id") != c.id || json_string(rec, "kind") != "low_delay_asym" ||
        num("k") != c.k || num("hop") != c.hop || num("support") != c.support || num("limiter_sub") != c.limiter_sub ||
        num("ramp_samples") != c.ramp_samples)
        throw std::runtime_error(where + ": contract record disagrees with " + c.id);
    if (json_string(rec, "window_hash") != window_round_hash(c))
        throw std::runtime_error(where + ": contract window hash differs from the native windows");
    if (cfg.frontend.ref_policy && cfg.frontend.ramp_samples != c.ramp_samples)
        throw std::runtime_error("frontend ramp_samples differs from the contract's");
}

Engine::Engine(const std::string& onnx, const std::string& expected, const EngineConfig& cfg_)
    : c(parse_contract_id(expected)), cfg(cfg_), fe(c, cfg_.frontend), val(c), an_p(c), an_r(c), syn(c) {
    ort = std::make_unique<OrtStep>(onnx, cfg);
    check_record(onnx, ort->meta, c, cfg);
    const int H = c.hop, B = c.bins();
    fp.assign(H, 0.0f); fr.assign(H, 0.0f); fv.assign(H, 1);
    P.assign(B, {}); R.assign(B, {}); Y.assign(B, {});
    bline.assign(c.release_lead(), 0.0f); bout.assign(H, 0.0f);
    reset();
}

Engine::~Engine() = default;

int Engine::state_floats() const { return static_cast<int>(ort->S); }
std::string Engine::graph_contract_hash() const { return ort->meta[1]; }
std::vector<float>& Engine::neural_state() { return ort->state(); }
int Engine::bypass_hops() const { return (c.history() + c.hop - 1) / c.hop; }

void Engine::reset() {
    fe.reset(); val.reset(); an_p.reset(); an_r.reset(); syn.reset(); ort->reset();
    std::fill(bline.begin(), bline.end(), 0.0f);
    bypass_left = 0; xfade_pending = false;
    hops = 0; discontinuity_hops = 0; recoveries = 0;
}

void Engine::begin_recovery() {
    fe.reset(); val.reset(); an_p.reset(); an_r.reset(); syn.reset(); ort->reset();
    bypass_left = bypass_hops();
    xfade_pending = true;
    ++recoveries;
}

static inline void pack_spec(float* s, const std::complex<float>* p, const std::complex<float>* r, int B) {
    for (int b = 0; b < B; ++b) {
        s[b] = p[b].real(); s[B + b] = p[b].imag(); s[2 * B + b] = r[b].real(); s[3 * B + b] = r[b].imag();
    }
}

void Engine::process_hop(const float* prim, const float* ref, const uint8_t* avail, float* out) {
    const int H = c.hop, B = c.bins();
    const bool tm = cfg.timing;
    uint64_t t0 = tm ? now_ns() : 0, t1 = 0, t2 = 0, t3 = 0;
    const bool disc = fe.process(prim, ref, avail, fp.data(), fr.data(), fv.data());
    const float v = val.push(fv.data());
    if (tm) t1 = now_ns();
    an_p.push(fp.data(), P.data());
    an_r.push(fr.data(), R.data());
    if (tm) t2 = now_ns();
    const float* step_out = nullptr;
    if (disc) {                                   // never into the recurrent model
        ort->reset();
        std::copy(P.begin(), P.end(), Y.begin());
        ++discontinuity_hops;
    } else {
        pack_spec(ort->spec.data(), P.data(), R.data(), B);
        ort->valid[0] = v;
        ort->run();
        step_out = ort->out.data();
        for (int b = 0; b < B; ++b) Y[b] = {step_out[b], step_out[B + b]};
    }
    if (tm) t3 = now_ns();
    syn.push(Y.data(), out);
    // bypass: the frontend primary delayed by L - H samples (always tracked, released only in recovery)
    if (bypass_left > 0 || xfade_pending) {
        const int D = c.release_lead();
        for (int i = 0; i < H; ++i) bout[i] = i < D ? bline[i] : fp[i - D];
        if (disc) std::fill(bout.begin(), bout.end(), 0.0f);          // invalid primary: finite silence
        if (bypass_left > 0) {
            std::copy(bout.begin(), bout.end(), out);
            --bypass_left;
        } else {
            for (int i = 0; i < H; ++i) {
                const float w = static_cast<float>(i + 1) / static_cast<float>(H);
                out[i] = (1.0f - w) * bout[i] + w * out[i];
            }
            xfade_pending = false;
        }
    }
    {
        const int D = c.release_lead();
        std::memcpy(bline.data(), fp.data() + (H - D), sizeof(float) * D);
    }
    ++hops;
    if (tm) {
        const uint64_t t4 = now_ns();
        stage_ns.frontend = t1 - t0; stage_ns.analysis = t2 - t1; stage_ns.step = t3 - t2;
        stage_ns.synthesis = t4 - t3; stage_ns.total = t4 - t0;
    }
    if (trace) {
        Trace& t = *trace;
        t.frontend_out.assign(fp.begin(), fp.end());
        t.frontend_out.insert(t.frontend_out.end(), fr.begin(), fr.end());
        t.frame_valid = v;
        t.p.assign(P.begin(), P.end()); t.r.assign(R.begin(), R.end());
        t.step_out.assign(2 * B, 0.0f);
        if (step_out)
            for (int b = 0; b < B; ++b) { t.step_out[2 * b] = step_out[b]; t.step_out[2 * b + 1] = step_out[B + b]; }
        t.discontinuity = disc;
    }
}

void Engine::flush_hop(float* out) {
    const int H = c.hop, B = c.bins();
    std::fill(fp.begin(), fp.end(), 0.0f);
    std::fill(fr.begin(), fr.end(), 0.0f);
    std::fill(fv.begin(), fv.end(), 1);
    const float v = val.push(fv.data());
    an_p.push(fp.data(), P.data());
    an_r.push(fr.data(), R.data());
    pack_spec(ort->spec.data(), P.data(), R.data(), B);
    ort->valid[0] = v;
    ort->run();
    const float* so = ort->out.data();
    for (int b = 0; b < B; ++b) Y[b] = {so[b], so[B + b]};
    syn.push(Y.data(), out);
    ++hops;
    (void)H;
    if (trace) {
        Trace& t = *trace;
        t.frontend_out.assign(2 * c.hop, 0.0f);
        t.frame_valid = v;
        t.p.assign(P.begin(), P.end()); t.r.assign(R.begin(), R.end());
        t.step_out.assign(2 * B, 0.0f);
        for (int b = 0; b < B; ++b) { t.step_out[2 * b] = so[b]; t.step_out[2 * b + 1] = so[B + b]; }
        t.discontinuity = false;
    }
}

// ---- state blob ---------------------------------------------------------------------------------------------------
namespace {
struct Writer {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
    template <class T> void pod(const T& v) { raw(&v, sizeof(T)); }
    void str(const std::string& s) { pod<uint32_t>(static_cast<uint32_t>(s.size())); raw(s.data(), s.size()); }
    template <class T> void vec(const std::vector<T>& v) { pod<uint64_t>(v.size()); raw(v.data(), v.size() * sizeof(T)); }
};
struct Reader {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    void raw(void* p, size_t n) {
        if (at + n > b.size()) throw std::runtime_error("stream state blob is truncated");
        std::memcpy(p, b.data() + at, n); at += n;
    }
    template <class T> T pod() { T v; raw(&v, sizeof(T)); return v; }
    std::string str() { std::string s(pod<uint32_t>(), '\0'); raw(&s[0], s.size()); return s; }
    // into an existing vector of the same length only: a state never changes a buffer's size
    template <class T> void vec(std::vector<T>& v, const char* what) {
        if (pod<uint64_t>() != v.size()) throw std::runtime_error(std::string("stream state: ") + what + " size differs");
        raw(v.data(), v.size() * sizeof(T));
    }
};
const char MAGIC[] = "VLDS1";
}  // namespace

std::string Engine::config_tag() const {
    return "limiter=" + std::to_string(cfg.frontend.limiter) + ";ref_policy=" + std::to_string(cfg.frontend.ref_policy) +
           ";ramp=" + std::to_string(cfg.frontend.ramp_samples) + ";S=" + std::to_string(ort->S) +
           ";graph=" + ort->meta[1];
}

std::vector<uint8_t> Engine::save_state() const {
    Writer w;
    w.raw(MAGIC, sizeof(MAGIC));
    w.str(c.id); w.str(window_round_hash(c)); w.str(config_tag());
    const LimiterState& l = fe.lim.st;
    w.pod<uint8_t>(l.has_env); w.pod(l.env); w.pod(l.floor); w.pod(l.gain); w.pod(l.engaged); w.pod(l.run);
    w.pod(fe.since); w.pod(fe.sample); w.pod(fe.discontinuities); w.pod<uint8_t>(fe.prev_avail);
    w.pod(val.last_bad); w.pod(val.sample);
    w.vec(an_p.hist); w.vec(an_r.hist); w.vec(syn.pending);
    w.vec(ort->st[ort->cur]);
    w.vec(bline); w.pod<int32_t>(bypass_left); w.pod<uint8_t>(xfade_pending);
    w.pod(hops); w.pod(discontinuity_hops); w.pod(recoveries);
    return w.b;
}

void Engine::load_state(const std::vector<uint8_t>& blob) {
    Reader r{blob};
    char magic[sizeof(MAGIC)];
    r.raw(magic, sizeof(MAGIC));
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) throw std::runtime_error("not a vaani_ld stream state");
    if (r.str() != c.id) throw std::runtime_error("stream state belongs to another audio contract");
    if (r.str() != window_round_hash(c)) throw std::runtime_error("stream state was built under other windows");
    if (r.str() != config_tag()) throw std::runtime_error("stream state was built under another configuration");
    // parse into temporaries first: a refused blob leaves the engine untouched
    LimiterState l;
    l.has_env = r.pod<uint8_t>(); l.env = r.pod<double>(); l.floor = r.pod<double>(); l.gain = r.pod<double>();
    l.engaged = r.pod<int64_t>(); l.run = r.pod<int64_t>();
    const int64_t since = r.pod<int64_t>(), sample = r.pod<int64_t>(), discs = r.pod<int64_t>();
    const bool prev = r.pod<uint8_t>();
    const int64_t last_bad = r.pod<int64_t>(), vsample = r.pod<int64_t>();
    auto hp = an_p.hist, hr = an_r.hist, pend = syn.pending;
    r.vec(hp, "analysis history"); r.vec(hr, "analysis history"); r.vec(pend, "synthesis overlap");
    std::vector<float> st(ort->S);
    r.vec(st, "neural state");
    auto bl = bline;
    r.vec(bl, "bypass delay line");
    const int32_t bleft = r.pod<int32_t>();
    const bool xf = r.pod<uint8_t>();
    const int64_t h = r.pod<int64_t>(), dh = r.pod<int64_t>(), rec = r.pod<int64_t>();
    if (r.at != blob.size()) throw std::runtime_error("stream state blob has trailing bytes");
    fe.lim.st = l; fe.since = since; fe.sample = sample; fe.discontinuities = discs; fe.prev_avail = prev;
    val.last_bad = last_bad; val.sample = vsample;
    an_p.hist = hp; an_r.hist = hr; syn.pending = pend;
    ort->reset();
    ort->st[0] = st;
    bline = bl; bypass_left = bleft; xfade_pending = xf;
    hops = h; discontinuity_hops = dh; recoveries = rec;
}

// ---- 48 kHz runner ------------------------------------------------------------------------------------------------
Engine48::Engine48(Engine& e_, const Fir& fir_)
    : e(e_), fir(fir_), dec(2, fir_.h, 3 * e_.contract().hop), itp(1, fir_.h) {
    const int H = e.contract().hop;
    in48.assign(2 * 3 * H, 0.0f); x16.assign(2 * H, 0.0f); y16.assign(H, 0.0f); av16.assign(H, 1);
}

void Engine48::reset() { e.reset(); dec.reset(); itp.reset(); }

void Engine48::process(const float* prim48, const float* ref48, const uint8_t* avail48, float* out48) {
    const int H = e.contract().hop, N = 3 * H;
    const uint64_t s0 = now_ns();
    bool prim_bad = false;
    for (int i = 0; i < N; ++i) {
        const float p = prim48[i], r = ref48[i];
        if (!std::isfinite(p)) prim_bad = true;
        in48[i] = std::isfinite(p) ? p : 0.0f;      // the FIR must not smear a NaN over its span
        in48[N + i] = std::isfinite(r) ? r : 0.0f;
    }
    for (int m = 0; m < H; ++m) av16[m] = avail48[3 * m] && avail48[3 * m + 1] && avail48[3 * m + 2];
    dec.process(in48.data(), N, x16.data());
    if (prim_bad) x16[0] = std::nanf("");           // keep the discontinuity visible to the frontend
    const uint64_t s1 = now_ns();
    e.process_hop(x16.data(), x16.data() + H, av16.data(), y16.data());
    const uint64_t s2 = now_ns();
    itp.process(y16.data(), H, out48);
    const uint64_t s3 = now_ns();
    e.stage_ns.resample = (s1 - s0) + (s3 - s2);
}

void Engine48::flush(float* out48) {
    e.flush_hop(y16.data());
    itp.process(y16.data(), e.contract().hop, out48);
}

std::vector<uint8_t> Engine48::save_state() const {
    Writer w;
    w.b = e.save_state();
    w.str(fir.sha256);
    w.vec(dec.state); w.vec(itp.state);
    return w.b;
}

void Engine48::load_state(const std::vector<uint8_t>& blob) {
    // the engine's part is length-prefixed by its own layout: find it by saving a template of the same shape
    const size_t n = e.save_state().size();
    if (blob.size() < n) throw std::runtime_error("stream state blob is truncated");
    Reader r{blob, n};
    if (r.str() != fir.sha256) throw std::runtime_error("stream state was built with another resampler");
    auto ds = dec.state, is = itp.state;
    r.vec(ds, "decimator state"); r.vec(is, "interpolator state");
    if (r.at != blob.size()) throw std::runtime_error("stream state blob has trailing bytes");
    e.load_state(std::vector<uint8_t>(blob.begin(), blob.begin() + n));
    dec.state = ds; itp.state = is;
}

}  // namespace vld
