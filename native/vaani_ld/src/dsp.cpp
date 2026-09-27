// See include/vaani_ld/dsp.hpp. Compiled with -ffp-contract=off: a fused multiply-add would change the float32
// rounding the golden vectors pin (the limiter's pairwise sums, the window products, the ramp).
#include "vaani_ld/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "pocketfft_hdronly.h"
#include "vaani_ld/sha256.hpp"

namespace vld {

Contract parse_contract_id(const std::string& id) {
    static const std::regex re("vaanife_ld_asym512_h([0-9]+)_s([0-9]+)_v1");
    std::smatch m;
    if (!std::regex_match(id, m, re))
        throw std::runtime_error("not a low-delay audio contract id: " + id);
    Contract c;
    c.id = id;
    c.hop = std::stoi(m[1]);
    c.support = std::stoi(m[2]);
    if (!(0 < c.hop && c.hop < c.support && c.support <= 2 * c.hop) || c.hop % c.limiter_sub)
        throw std::runtime_error("inconsistent contract " + id);
    return c;
}

void ld_windows(const Contract& c, std::vector<double>& a, std::vector<double>& p, std::vector<double>& s) {
    const int K = c.k, H = c.hop, L = c.support, X = L - H;
    const double pi = M_PI;
    a.assign(K, 0.0); p.assign(K, 0.0); s.assign(K, 0.0);
    for (int n = 0; n < K - H; ++n) a[n] = std::sin(pi * n / (2.0 * (K - H)));
    for (int n = 0; n < H; ++n) a[K - H + n] = std::cos(pi * n / (2.0 * H));
    for (int n = K - L; n < K; ++n) p[n] = 1.0;
    for (int n = 0; n < X; ++n) {
        double sn = std::sin(pi * n / (2.0 * X)), cs = std::cos(pi * n / (2.0 * X));
        p[K - L + n] = sn * sn;
        p[K - X + n] = cs * cs;
    }
    for (int n = 0; n < K; ++n) s[n] = a[n] > 0 ? p[n] / a[n] : 0.0;
}

// ---- limiter ----------------------------------------------------------------------------------------
static const double HEADROOM_DB = 26.0, RELEASE_MS = 50.0, ENV_MS = 500.0, FAR_FIELD_DB = 4.0;
static const double FLOOR_UP = 0.002, FLOOR_DOWN = 0.3, FLOOR_MAX = 3.0, MIN_ENV = 1e-4;

Limiter::Limiter(int sub_, int sr) : sub(sub_) {
    thr = std::pow(10.0, HEADROOM_DB / 20.0);
    rel = 1.0 - std::exp(-sub / (RELEASE_MS * 1e-3 * sr));
    env_a = 1.0 - std::exp(-sub / (ENV_MS * 1e-3 * sr));
    far = FAR_FIELD_DB; floor_up = FLOOR_UP; floor_down = FLOOR_DOWN; floor_max = FLOOR_MAX; min_env = MIN_ENV;
}

// float32 mean of x[a:b]^2 in NumPy's pairwise order (8 accumulators for 8 <= n <= 128)
static float pairwise_sq_mean(const float* x, int a, int b) {
    const int n = b - a;
    float res;
    if (n < 8) {
        res = 0.0f;
        for (int i = a; i < b; ++i) { float v = x[i] * x[i]; res = res + v; }
    } else {
        float r[8];
        for (int q = 0; q < 8; ++q) r[q] = x[a + q] * x[a + q];
        const int m = n - n % 8;
        for (int i = 8; i < m; i += 8)
            for (int q = 0; q < 8; ++q) { float v = x[a + i + q] * x[a + i + q]; r[q] += v; }
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (int i = a + m; i < b; ++i) { float v = x[i] * x[i]; res += v; }
    }
    return res / static_cast<float>(n);
}

void Limiter::process(float* p, float* r, int n) {
    const float eps = 1e-10f;
    for (int a = 0; a < n; a += sub) {
        const int b = std::min(a + sub, n);
        const double ep = static_cast<double>(pairwise_sq_mean(p, a, b) + eps);
        const double er = static_cast<double>(pairwise_sq_mean(r, a, b) + eps);
        float pmax = 0.0f;
        for (int i = a; i < b; ++i) {
            pmax = std::max(pmax, std::fabs(p[i]));
            pmax = std::max(pmax, std::fabs(r[i]));
        }
        const double ratio_db = 10.0 * std::log10(ep / er);
        const double rate = ratio_db > st.floor ? floor_up : floor_down;
        st.floor = std::min(st.floor + rate * (ratio_db - st.floor), floor_max);
        const double rms = std::sqrt(0.5 * (ep + er));
        if (!st.has_env) { st.env = rms; st.has_env = true; }
        const double peak = static_cast<double>(pmax) + 1e-9;
        const double ceiling = thr * std::max(st.env, min_env);
        const bool hit = peak > ceiling && ratio_db <= st.floor + far;
        if (hit) {
            st.gain = std::min(st.gain, ceiling / peak);
            ++st.engaged;
        } else {
            st.gain += rel * (1.0 - st.gain);
            if (st.gain > 0.999) st.gain = 1.0;
            st.env += env_a * (rms - st.env);
        }
        const float g32 = static_cast<float>(st.gain);
        for (int i = a; i < b; ++i) { p[i] = p[i] * g32; r[i] = r[i] * g32; }
    }
}

// ---- frontend ------------------------------------------------------------------------------------------
Frontend::Frontend(const Contract& c_, const FrontendConfig& cfg_) : lim(c_.limiter_sub), c(c_), cfg(cfg_) { reset(); }

void Frontend::reset() {
    lim.reset();
    since = cfg.ref_policy ? cfg.ramp_samples : 0;
    prev_avail = true; sample = 0; discontinuities = 0;
}

bool Frontend::process(const float* prim, const float* ref, const uint8_t* avail, float* op, float* orr, uint8_t* valid) {
    const int H = c.hop;
    bool disc = false;
    for (int i = 0; i < H; ++i) {
        bool av = avail[i] != 0;
        float r = ref[i];
        if (!std::isfinite(r)) { av = false; r = 0.0f; }
        float p = prim[i];
        if (!std::isfinite(p)) { disc = true; p = 0.0f; }
        if (cfg.ref_policy && !av) r = 0.0f;
        op[i] = p; orr[i] = r; valid[i] = av ? 1 : 0;
    }
    if (disc) ++discontinuities;
    if (cfg.limiter) { lim.process(op, orr, H); lim.st.engaged = 0; }
    if (cfg.ref_policy) {   // pipeline.ref_gain_step, sequentially
        const int n_ramp = cfg.ramp_samples;
        int64_t cnt = since;
        for (int i = 0; i < H; ++i) {
            const bool av = valid[i] != 0;
            const bool prev = i == 0 ? prev_avail : valid[i - 1] != 0;
            if (av && !prev) cnt = 0;
            else if (i > 0) cnt = cnt + 1;
            const float k = static_cast<float>(cnt + 1) / static_cast<float>(n_ramp);
            const float g = av ? (cnt < n_ramp ? k : 1.0f) : 0.0f;
            orr[i] = orr[i] * g;
        }
        since = std::min<int64_t>(cnt + 1, n_ramp);
        prev_avail = valid[H - 1] != 0;
    }
    sample += H;
    return disc;
}

float Validity::push(const uint8_t* avail) {
    for (int i = c.hop - 1; i >= 0; --i)
        if (!avail[i]) { last_bad = sample + i; break; }
    sample += c.hop;
    return last_bad < sample - c.k ? 1.0f : 0.0f;
}

// ---- analysis / synthesis ---------------------------------------------------------------------------------
Analyzer::Analyzer(const Contract& c_) : c(c_) {
    std::vector<double> a64, p64, s64;
    ld_windows(c, a64, p64, s64);
    a.assign(a64.begin(), a64.end());
    frame.assign(c.k, 0.0f);
    reset();
}

void Analyzer::reset() { hist.assign(c.history(), 0.0f); }

void Analyzer::push(const float* hop, std::complex<float>* spec) {
    const int K = c.k, H = c.hop, Hs = c.history();
    std::memcpy(frame.data(), hist.data(), sizeof(float) * Hs);
    std::memcpy(frame.data() + Hs, hop, sizeof(float) * H);
    std::memcpy(hist.data(), frame.data() + H, sizeof(float) * Hs);
    for (int n = 0; n < K; ++n) frame[n] = frame[n] * a[n];
    pocketfft::r2c<float>({static_cast<size_t>(K)}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0, true,
                          frame.data(), spec, 1.0f, 1);
}

Synthesizer::Synthesizer(const Contract& c_) : c(c_) {
    std::vector<double> a64, p64, s64;
    ld_windows(c, a64, p64, s64);
    s_tail.assign(s64.begin() + (c.k - c.support), s64.end());
    y.assign(c.k, 0.0f);
    reset();
}

void Synthesizer::reset() { pending.assign(c.crossfade(), 0.0f); }

void Synthesizer::push(const std::complex<float>* spec, float* out) {
    const int K = c.k, H = c.hop, L = c.support, X = c.crossfade();
    pocketfft::c2r<float>({static_cast<size_t>(K)}, {sizeof(std::complex<float>)}, {sizeof(float)}, 0, false,
                          spec, y.data(), 1.0f / K, 1);
    float* t = y.data() + (K - L);
    for (int n = 0; n < L; ++n) t[n] = t[n] * s_tail[n];
    for (int n = 0; n < X; ++n) t[n] += pending[n];
    std::memcpy(pending.data(), t + H, sizeof(float) * X);
    std::memcpy(out, t, sizeof(float) * H);
}

// ---- resamplers ----------------------------------------------------------------------------------------------
std::string coef_sha256(const std::vector<double>& h) {
    static_assert(sizeof(double) == 8, "float64 coefficients");
    return sha256_hex(reinterpret_cast<const uint8_t*>(h.data()), h.size() * sizeof(double));   // little-endian hosts
}

static std::string json_field(const std::string& js, const std::string& key) {
    const std::string k = "\"" + key + "\"";
    size_t at = js.find(k);
    if (at == std::string::npos) throw std::runtime_error("coefficient file lacks \"" + key + "\"");
    return js.substr(js.find(':', at) + 1);
}

Fir load_fir(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    const std::string js = ss.str();
    Fir fir;
    std::string idv = json_field(js, "id");
    fir.id = idv.substr(idv.find('"') + 1, idv.find('"', idv.find('"') + 1) - idv.find('"') - 1);
    std::string co = json_field(js, "coefficients");
    const char* p = co.c_str() + co.find('[') + 1;
    while (true) {
        char* end;
        double v = std::strtod(p, &end);
        if (end == p) break;
        fir.h.push_back(v);
        p = end;
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == ']') break;
    }
    std::string sh = json_field(js, "sha256");
    fir.sha256 = sh.substr(sh.find('"') + 1, 64);
    std::string pk = json_field(js, "pair_peak_ms");
    fir.pair_peak_ms = std::strtod(pk.c_str(), nullptr);
    if (fir.h.empty()) throw std::runtime_error(path + ": no coefficients");
    if (coef_sha256(fir.h) != fir.sha256)
        throw std::runtime_error(path + ": coefficient sha256 differs from the recorded one");
    return fir;
}

Decimate3::Decimate3(int channels, const std::vector<double>& h_) : ch(channels), h(h_) { reset(); }

void Decimate3::reset() { state.assign(static_cast<size_t>(ch) * (h.size() - 1), 0.0); }

void Decimate3::process(const float* in, int n, float* out) {
    if (n % 3) throw std::runtime_error("Decimate3 needs blocks whose length is a multiple of 3");
    const int T = static_cast<int>(h.size()) - 1;
    buf.resize(static_cast<size_t>(T + n));
    for (int c = 0; c < ch; ++c) {
        double* s = state.data() + static_cast<size_t>(c) * T;
        std::memcpy(buf.data(), s, sizeof(double) * T);
        for (int i = 0; i < n; ++i) buf[T + i] = static_cast<double>(in[static_cast<size_t>(c) * n + i]);
        for (int m = 0; m < n / 3; ++m) {
            const int t = T + 3 * m;            // buf index of input sample 3m of this block
            double acc = 0.0;
            for (int k = 0; k <= T; ++k) acc += h[k] * buf[t - k];
            out[static_cast<size_t>(c) * (n / 3) + m] = static_cast<float>(acc);
        }
        std::memcpy(s, buf.data() + n, sizeof(double) * T);
    }
}

Interpolate3::Interpolate3(int channels, const std::vector<double>& h) : ch(channels) {
    h3.resize(h.size());
    for (size_t i = 0; i < h.size(); ++i) h3[i] = 3.0 * h[i];
    hist = static_cast<int>((h.size() - 1) / 3) + 1;
    reset();
}

void Interpolate3::reset() { state.assign(static_cast<size_t>(ch) * hist, 0.0); }

void Interpolate3::process(const float* in, int n, float* out) {
    const int taps = static_cast<int>(h3.size());
    for (int c = 0; c < ch; ++c) {
        double* s = state.data() + static_cast<size_t>(c) * hist;   // s[j] = x[m - 1 - j] relative to the block
        for (int m = 0; m < n; ++m) {
            // shift the new sample in: s[0] is the current input x[m]
            std::memmove(s + 1, s, sizeof(double) * (hist - 1));
            s[0] = static_cast<double>(in[static_cast<size_t>(c) * n + m]);
            for (int ph = 0; ph < 3; ++ph) {
                double acc = 0.0;
                for (int j = 0; 3 * j + ph < taps; ++j) acc += h3[3 * j + ph] * s[j];
                out[static_cast<size_t>(c) * 3 * n + 3 * m + ph] = static_cast<float>(acc);
            }
        }
    }
}

}  // namespace vld
