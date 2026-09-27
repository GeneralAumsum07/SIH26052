// Native runtime unit tests (Task 7): state continuation, contract refusal, reset determinism, interleaved streams,
// the release-timeline simulation, zero host-side DSP allocations and the recovery bypass.
//   vld_unit_test MODELS_DIR RESAMPLER_JSON_DIR
// MODELS_DIR holds <contract>/model.onnx (the Task 6 golden graphs). Exits 1 on any failure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "vaani_ld/engine.hpp"
#include "vaani_ld/sim.hpp"

static uint64_t g_news = 0;
void* operator new(size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

using namespace vld;

static int failures = 0, passes = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (cond) { ++passes; }                                                  \
        else { ++failures; std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } \
    } while (0)

static std::string MODELS, RJSON;
static const std::string A = "vaanife_ld_asym512_h96_s160_v1", B = "vaanife_ld_asym512_h96_s144_v1";
static std::string model(const std::string& c) { return MODELS + "/" + c + "/model.onnx"; }

struct Signal { std::vector<float> p, r; std::vector<uint8_t> av; };
static Signal signal(int n, uint32_t seed, float level = 0.1f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, level);
    Signal s{std::vector<float>(n), std::vector<float>(n), std::vector<uint8_t>(n, 1)};
    for (int i = 0; i < n; ++i) {
        s.p[i] = 0.3f * std::sin(2.0f * 3.14159265f * 150.0f * i / 16000.0f) + nd(g);
        s.r[i] = nd(g);
    }
    return s;
}

static std::vector<float> run(Engine& e, const Signal& s, int from_hop, int to_hop) {
    const int H = e.contract().hop;
    std::vector<float> y(static_cast<size_t>(to_hop - from_hop) * H);
    for (int j = from_hop; j < to_hop; ++j)
        e.process_hop(&s.p[j * H], &s.r[j * H], &s.av[j * H], &y[static_cast<size_t>(j - from_hop) * H]);
    return y;
}

static bool throws(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

static void test_state_and_reset() {
    const int hops = 80, cut = 37;
    Engine e(model(A), A);
    const int H = e.contract().hop;
    const Signal s = signal(hops * H, 1);
    const auto full = run(e, s, 0, hops);
    // reset determinism
    e.reset();
    CHECK(run(e, s, 0, hops) == full, "reset() does not reproduce the stream bit for bit");
    // state continuation into a fresh engine
    e.reset();
    auto head = run(e, s, 0, cut);
    const auto blob = e.save_state();
    Engine f(model(A), A);
    f.load_state(blob);
    auto tail = run(f, s, cut, hops);
    head.insert(head.end(), tail.begin(), tail.end());
    CHECK(head == full, "save_state/load_state does not continue the stream bit for bit");
    CHECK(f.hops == hops, "restored hop counter");
    // refusal: another contract's state, a truncated blob, another configuration
    Engine other(model(B), B);
    CHECK(throws([&] { other.load_state(blob); }), "a state of another contract is accepted");
    auto cut_blob = blob;
    cut_blob.resize(blob.size() - 5);
    CHECK(throws([&] { f.load_state(cut_blob); }), "a truncated state is accepted");
    EngineConfig nl;
    nl.frontend.limiter = false;
    Engine g(model(A), A, nl);
    CHECK(throws([&] { g.load_state(blob); }), "a state of another configuration is accepted");
    // a refused blob leaves the engine untouched
    auto bad = blob;
    bad[bad.size() - 1] ^= 0;   // same bytes, then a size field corrupted:
    bad.insert(bad.end(), 1, 0);
    const auto before = f.save_state();
    CHECK(throws([&] { f.load_state(bad); }), "a blob with trailing bytes is accepted");
    CHECK(f.save_state() == before, "a refused blob changed the engine");
}

static void test_contract_refusal() {
    CHECK(throws([&] { Engine e(model(A), B); }), "a graph stamped for another contract is accepted");
    CHECK(throws([&] { parse_contract_id("vaanife_c0"); }), "a legacy contract id parses as low-delay");
    CHECK(throws([&] { parse_contract_id("vaanife_ld_asym512_h96_s200_v1"); }), "L > 2H is accepted");
    EngineConfig r;
    r.frontend.ramp_samples = 1000;
    CHECK(throws([&] { Engine e(model(A), A, r); }), "a frontend ramp other than the contract's is accepted");
}

static void test_interleaved() {
    const int hops = 60;
    Engine a(model(A), A), b(model(A), A);
    const int H = a.contract().hop;
    const Signal s1 = signal(hops * H, 2), s2 = signal(hops * H, 3, 0.02f);
    const auto y1 = run(a, s1, 0, hops), y2 = run(b, s2, 0, hops);
    a.reset(); b.reset();
    std::vector<float> z1(y1.size()), z2(y2.size());
    for (int j = 0; j < hops; ++j) {
        b.process_hop(&s2.p[j * H], &s2.r[j * H], &s2.av[j * H], &z2[j * H]);
        a.process_hop(&s1.p[j * H], &s1.r[j * H], &s1.av[j * H], &z1[j * H]);
    }
    CHECK(z1 == y1 && z2 == y2, "interleaved streams on one graph interfere");
}

static void test_no_dsp_alloc() {
    const Contract c = parse_contract_id(A);
    const int H = c.hop;
    Frontend fe(c, FrontendConfig{});
    Validity v(c);
    Analyzer an(c);
    Synthesizer sy(c);
    const Fir fir = load_fir(RJSON + "/r1_minphase_kaiser193_v1.json");
    Decimate3 dec(2, fir.h, 3 * H);
    Interpolate3 itp(1, fir.h);
    const Signal s = signal(200 * 3 * H, 4);
    std::vector<float> op(H), orr(H), y(H), x16(2 * H), in48(6 * H), y48(3 * H);
    std::vector<uint8_t> val(H);
    std::vector<std::complex<float>> P(c.bins());
    auto hop = [&](int j) {
        std::copy(&s.p[j * 3 * H], &s.p[(j + 1) * 3 * H], in48.begin());
        std::copy(&s.r[j * 3 * H], &s.r[(j + 1) * 3 * H], in48.begin() + 3 * H);
        dec.process(in48.data(), 3 * H, x16.data());
        fe.process(x16.data(), x16.data() + H, s.av.data(), op.data(), orr.data(), val.data());
        v.push(val.data());
        an.push(op.data(), P.data());
        sy.push(P.data(), y.data());
        itp.process(y.data(), H, y48.data());
    };
    for (int j = 0; j < 10; ++j) hop(j);           // PocketFFT builds its plan cache on first use
    const uint64_t n0 = g_news;
    for (int j = 10; j < 200; ++j) hop(j);
    CHECK(g_news == n0, "host-side DSP allocated after warmup");
    std::printf("dsp allocations after warmup: %llu\n", static_cast<unsigned long long>(g_news - n0));
    // the whole engine hop, inference included, is reported (the library's own behaviour is measured separately)
    Engine e(model(A), A);
    std::vector<float> out(H);
    for (int j = 0; j < 20; ++j) e.process_hop(&s.p[j * H], &s.r[j * H], &s.av[j * H], out.data());
    const uint64_t e0 = g_news;
    for (int j = 20; j < 200; ++j) e.process_hop(&s.p[j * H], &s.r[j * H], &s.av[j * H], out.data());
    std::printf("engine hop (ORT Run included) operator new calls per hop after warmup: %.2f\n",
                (g_news - e0) / 180.0);
}

static void test_bypass() {
    EngineConfig cfg;
    cfg.frontend.limiter = false;    // the bypass releases the frontend primary; without the limiter it is the input
    Engine e(model(A), A, cfg);
    const Contract& c = e.contract();
    const int H = c.hop, D = c.release_lead(), hops = 40, at = 12;
    Signal s = signal(hops * H, 5);
    std::vector<float> y(hops * H);
    for (int j = 0; j < hops; ++j) {
        if (j == at) e.begin_recovery();
        e.process_hop(&s.p[j * H], &s.r[j * H], &s.av[j * H], &y[j * H]);
    }
    const int nb = e.bypass_hops();
    CHECK(nb == (c.k - H + H - 1) / H, "bypass length is ceil((K - H) / H) hops");
    double err = 0;
    for (int j = at; j < at + nb; ++j)
        for (int i = 0; i < H; ++i) err = std::max(err, std::fabs(static_cast<double>(y[j * H + i]) - s.p[j * H + i - D]));
    CHECK(err == 0.0, "bypass is not the primary delayed by L - H samples");
    CHECK(e.recoveries == 1 && !e.in_recovery(), "recovery counted and finished after the crossfade hop");
    bool finite = true;
    for (float v : y) finite &= std::isfinite(v);
    CHECK(finite, "non-finite output around a recovery");
    // invalid primary during the bypass: finite silence for that hop
    Engine f(model(A), A, cfg);
    Signal t = s;
    t.p[(at + 1) * H + 7] = NAN;
    std::vector<float> z(hops * H);
    for (int j = 0; j < hops; ++j) {
        if (j == at) f.begin_recovery();
        f.process_hop(&t.p[j * H], &t.r[j * H], &t.av[j * H], &z[j * H]);
    }
    bool silent = true;
    for (int i = 0; i < H; ++i) silent &= z[(at + 1) * H + i] == 0.0f;
    CHECK(silent, "an invalid primary hop in the bypass is not silence");
    finite = true;
    for (float v : z) finite &= std::isfinite(v);
    CHECK(finite, "non-finite output with an invalid primary");
}

static void test_sim() {
    const Contract c = parse_contract_id(A);
    Timeline t;
    t.period = 48; t.hop_periods = 6; t.dproc_periods = 1; t.queue_periods = 2;
    Budget b;
    CHECK(check_timeline(t, c.hop, c.support, 0.333, &b).empty(), "1 ms periods, D_proc 1 ms refused");
    CHECK(t.delay_frames() == 3 * c.hop + 2 * 48, "delay = hop + D_proc + one playback period");
    CHECK(std::fabs(b.total_ms() - (10.0 + 0.333 + 2.0 + 0.6)) < 1e-9 && b.eligible(), "Section 4 budget (L 10, R1)");
    Budget b0;
    check_timeline(t, c.hop, c.support, 4.0, &b0);
    CHECK(!b0.eligible(), "R0 (the control) reported eligible");
    Timeline bad = t;
    bad.dproc_periods = 7;
    CHECK(!check_timeline(bad, c.hop, c.support, 0.4, nullptr).empty(), "D_proc > H accepted");
    bad = t; bad.period = 96; bad.hop_periods = 3;
    CHECK(!check_timeline(bad, c.hop, c.support, 0.4, nullptr).empty(), "2 ms periods accepted at L = 10 ms");
    const Contract c8 = parse_contract_id("vaanife_ld_asym512_h96_s128_v1");
    CHECK(check_timeline(bad, c8.hop, c8.support, 0.4, nullptr).empty(), "2 ms periods refused at L = 8 ms");
    bad = t; bad.period = 40; bad.hop_periods = 7;
    CHECK(!check_timeline(bad, c.hop, c.support, 0.4, nullptr).empty(), "a period that does not divide the hop accepted");

    const double dP = 48.0;
    for (int d = 1; d <= 2; ++d) {
        Timeline u = t;
        u.dproc_periods = d;
        const double lim = d * dP;
        for (auto prof : std::vector<std::function<double(int64_t)>>{
                 [](int64_t) { return 0.0; }, [lim](int64_t) { return 0.5 * lim; }, [lim](int64_t) { return lim; },
                 [lim](int64_t j) { return (j * 7919 % 97) / 96.0 * lim; }}) {
            const SimResult r = simulate(u, 5000, prof);
            CHECK(r.late_hops == 0 && r.late_periods == 0 && r.stale_periods == 0 && r.misplaced == 0,
                  "on-time processing produced late, stale or misplaced periods");
            CHECK(r.played_ok == 5000 * u.hop_periods, "not every output period played at its time");
            CHECK(r.max_ring <= u.hop_periods + d, "ring occupancy above n + d periods");
            CHECK(r.silence_start_periods == u.delay_frames() / u.period, "startup silence differs from the delay");
        }
        // late processing: every 100th hop misses its deadline by half a period; it is dropped and recovered
        const SimResult r = simulate(u, 5000, [lim](int64_t j) { return j % 100 == 99 ? lim + 24.0 : 0.3 * lim; });
        CHECK(r.late_hops == 50 && r.recoveries == 50, "late hops not detected");
        CHECK(r.late_periods == 50 * u.hop_periods && r.misplaced == 0, "a late hop's periods not replaced by silence");
        CHECK(r.played_ok == (5000 - 50) * u.hop_periods, "on-time hops after a late one not played");
        CHECK(r.max_ring <= u.ring_periods(), "ring above its capacity");
        // processing slower than real time: a growing backlog, every hop late, the ring still bounded
        const SimResult s = simulate(u, 500, [](int64_t) { return 400.0; });
        CHECK(s.late_hops > 400 && s.max_ring <= u.ring_periods() && s.misplaced == 0, "overload not bounded");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s MODELS_DIR RESAMPLER_JSON_DIR\n", argv[0]); return 2; }
    MODELS = argv[1]; RJSON = argv[2];
    try {
        test_contract_refusal();
        test_state_and_reset();
        test_interleaved();
        test_no_dsp_alloc();
        test_bypass();
        test_sim();
    } catch (const std::exception& ex) {
        std::printf("FAIL exception: %s\n", ex.what());
        return 1;
    }
    std::printf("{\"passes\": %d, \"failures\": %d}\n", passes, failures);
    return failures ? 1 : 0;
}
