// vaani_ld_run: the native low-delay runtime (low-delay plan Task 7).
//
//   vaani_ld_run info     --model M.onnx --contract ID
//   vaani_ld_run wav      --model M.onnx --contract ID --in IN.wav --out OUT.wav [--resampler R.json] [--avail A.wav]
//                         [--split-at HOP] [--no-limiter] [--no-ref-policy]
//   vaani_ld_run simulate --contract ID [--period 48] [--dproc 1] [--queue 2] [--hops 20000]
//                         [--proc-ms X | --profile early|nominal|late|jitter] [--resampler R.json]
//   vaani_ld_run live     --model M.onnx --contract ID --resampler R.json --dproc D [--period 48] [--queue 2]
//                         [--capture hw:0,0] [--playback hw:0,0] [--duration S] [--cpu N] [--prio 80]
//                         [--ctl "NAME=VALUE"] [--prim-ch 0] [--ref-ch 1] [--report R.json] [--allow-ineligible]
//
// WAV mode takes a 2-channel file (primary, reference) at 16 kHz, or at 48 kHz with --resampler, and writes the
// output aligned like LowDelayStreamEngine.run (the first L - H released samples dropped, the input's length).
// --split-at saves the whole stream state after that hop, builds a new engine, restores it and continues (the
// output must not change). Every mode prints one JSON object on stdout.
#include <sys/mman.h>
#include <sys/utsname.h>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vaani_ld/engine.hpp"
#include "vaani_ld/sim.hpp"
#include "vaani_ld/stats.hpp"
#include "vaani_ld/wav.hpp"
#ifdef VLD_WITH_ALSA
#include "vaani_ld/alsa_io.hpp"
#endif

using namespace vld;

namespace {

struct Args {
    std::string mode;
    std::map<std::string, std::string> kv;
    std::vector<std::string> ctl;
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    std::string need(const std::string& k) const {
        if (!has(k)) throw std::runtime_error("--" + k + " is required");
        return get(k);
    }
    double num(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
};

const char* FLAGS[] = {"no-limiter", "no-ref-policy", "allow-ineligible", "allow-plugin", "no-rt"};

Args parse(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: vaani_ld_run info|wav|simulate|live [options] (see src/main.cpp)");
    Args a;
    a.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + k);
        k = k.substr(2);
        bool flag = false;
        for (const char* f : FLAGS) flag |= k == f;
        if (flag) { a.kv[k] = "1"; continue; }
        if (i + 1 >= argc) throw std::runtime_error("--" + k + " needs a value");
        if (k == "ctl") a.ctl.push_back(argv[++i]);
        else a.kv[k] = argv[++i];
    }
    return a;
}

std::string q(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

EngineConfig engine_config(const Args& a) {
    EngineConfig cfg;
    cfg.frontend.limiter = !a.has("no-limiter");
    cfg.frontend.ref_policy = !a.has("no-ref-policy");
    cfg.timing = true;
    return cfg;
}

std::string kernel() {
    utsname u{};
    uname(&u);
    return std::string(u.sysname) + " " + u.release + " " + u.version + " " + u.machine;
}

// ---- info ----------------------------------------------------------------------------------------------------------
int run_info(const Args& a) {
    Engine e(a.need("model"), a.need("contract"), engine_config(a));
    const Contract& c = e.contract();
    std::printf("{\"contract\": %s, \"graph_contract_hash\": %s, \"window_hash\": %s, \"k\": %d, \"hop\": %d, "
                "\"support\": %d, \"release_lead\": %d, \"state_floats\": %d, \"bypass_hops\": %d, \"kernel\": %s}\n",
                q(c.id).c_str(), q(e.graph_contract_hash()).c_str(), q(window_round_hash(c)).c_str(), c.k, c.hop,
                c.support, c.release_lead(), e.state_floats(), e.bypass_hops(), q(kernel()).c_str());
    return 0;
}

// ---- wav -----------------------------------------------------------------------------------------------------------
int run_wav(const Args& a) {
    const std::string model = a.need("model"), cid = a.need("contract");
    const EngineConfig cfg = engine_config(a);
    Wav in = read_wav(a.need("in"));
    if (in.channels != 2) throw std::runtime_error("WAV mode takes 2 channels (primary, reference)");
    const bool r48 = a.has("resampler");
    std::unique_ptr<Fir> fir;
    if (r48) fir = std::make_unique<Fir>(load_fir(a.get("resampler")));
    const int want_sr = r48 ? 48000 : 16000;
    if (in.sr != want_sr) throw std::runtime_error("input is " + std::to_string(in.sr) + " Hz; this mode takes " + std::to_string(want_sr));
    std::vector<uint8_t> avail(in.frames(), 1);
    if (a.has("avail")) {
        Wav av = read_wav(a.get("avail"));
        if (av.frames() != in.frames()) throw std::runtime_error("--avail length differs from --in");
        for (size_t i = 0; i < avail.size(); ++i) avail[i] = av.ch[0][i] > 0.5f;
    }
    auto e = std::make_unique<Engine>(model, cid, cfg);
    std::unique_ptr<Engine48> e48;
    if (r48) e48 = std::make_unique<Engine48>(*e, *fir);
    const int H = e->contract().hop, rate = r48 ? 3 : 1, N = H * rate;
    const size_t n = in.frames(), hops = (n + N - 1) / N;
    const int64_t split = a.has("split-at") ? std::stoll(a.get("split-at")) : -1;
    std::vector<float> p(N), r(N), y((hops + 1) * N);
    std::vector<uint8_t> av(N);
    flush_denormals(false);    // WAV mode compares with the Python reference, which runs without FTZ
    LatencyHist total;
    for (size_t j = 0; j < hops; ++j) {
        for (int i = 0; i < N; ++i) {
            const size_t s = j * N + i;
            p[i] = s < n ? in.ch[0][s] : 0.0f; r[i] = s < n ? in.ch[1][s] : 0.0f; av[i] = s < n ? avail[s] : 1;
        }
        if (r48) e48->process(p.data(), r.data(), av.data(), &y[j * N]);
        else e->process_hop(p.data(), r.data(), av.data(), &y[j * N]);
        total.add(e->stage_ns.total + (r48 ? e->stage_ns.resample : 0));
        if (static_cast<int64_t>(j) == split) {       // state restart: a new engine resumes from the saved blob
            const std::vector<uint8_t> blob = r48 ? e48->save_state() : e->save_state();
            e48.reset();
            e = std::make_unique<Engine>(model, cid, cfg);
            if (r48) { e48 = std::make_unique<Engine48>(*e, *fir); e48->load_state(blob); }
            else e->load_state(blob);
        }
    }
    if (r48) e48->flush(&y[hops * N]);
    else e->flush_hop(&y[hops * N]);
    const size_t lead = static_cast<size_t>(e->contract().release_lead()) * rate;
    Wav out;
    out.sr = in.sr; out.channels = 1;
    out.ch.assign(1, std::vector<float>(y.begin() + lead, y.begin() + lead + n));
    write_wav(a.need("out"), out);
    std::printf("{\"mode\": \"wav\", \"contract\": %s, \"samples\": %zu, \"hops\": %zu, \"sr\": %d, \"resampler\": %s, "
                "\"discontinuity_hops\": %lld, \"split_at\": %lld, \"hop_ms\": %s}\n",
                q(cid).c_str(), n, hops + 1, in.sr, r48 ? q(fir->id).c_str() : "null",
                static_cast<long long>(e->discontinuity_hops), static_cast<long long>(split), total.json().c_str());
    return 0;
}

// ---- simulate ------------------------------------------------------------------------------------------------------------
Timeline timeline_of(const Args& a, const Contract& c) {
    Timeline t;
    t.period = static_cast<int>(a.num("period", 48));
    t.hop_periods = t.period > 0 ? 3 * c.hop / t.period : 0;
    t.dproc_periods = static_cast<int>(a.num("dproc", 1));
    t.queue_periods = static_cast<int>(a.num("queue", 2));
    return t;
}

std::string budget_json(const Budget& b, const std::vector<std::string>& why) {
    std::ostringstream o;
    o << "{\"support_ms\": " << b.support_ms << ", \"resampler_ms\": " << b.resampler_ms << ", \"io_ms\": " << b.io_ms
      << ", \"converters_ms\": " << b.converters_ms << ", \"fifo_ms\": " << b.fifo_ms << ", \"total_ms\": "
      << b.total_ms() << ", \"eligible\": " << (b.eligible() ? "true" : "false") << ", \"refused\": [";
    for (size_t i = 0; i < why.size(); ++i) o << (i ? ", " : "") << q(why[i]);
    return o.str() + "]}";
}

int run_simulate(const Args& a) {
    const Contract c = parse_contract_id(a.need("contract"));
    const Timeline t = timeline_of(a, c);
    const double pair = a.has("resampler") ? load_fir(a.get("resampler")).pair_peak_ms : a.num("pair-ms", 0.4);
    Budget b;
    const auto why = check_timeline(t, c.hop, c.support, pair, &b);
    const int64_t hops = static_cast<int64_t>(a.num("hops", 20000));
    const double P = t.period, dP = t.dproc_periods * P;
    const std::string prof = a.get("profile", "nominal");
    std::mt19937_64 g(static_cast<uint64_t>(a.num("seed", 0)));
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::function<double(int64_t)> proc;
    if (a.has("proc-ms")) { const double f = a.num("proc-ms", 0) * 48.0; proc = [f](int64_t) { return f; }; }
    else if (prof == "early") proc = [](int64_t) { return 0.0; };
    else if (prof == "nominal") proc = [dP](int64_t) { return 0.6 * dP; };
    else if (prof == "edge") proc = [dP](int64_t) { return dP; };
    else if (prof == "late") proc = [dP, P](int64_t j) { return j % 1000 == 999 ? dP + 0.5 * P : 0.5 * dP; };
    else if (prof == "jitter") proc = [dP, &g, &u](int64_t) { return u(g) * dP; };
    else throw std::runtime_error("unknown --profile " + prof);
    SimResult r;
    if (why.empty()) r = simulate(t, hops, proc);
    std::printf("{\"mode\": \"simulate\", \"contract\": %s, \"period\": %d, \"hop_periods\": %d, \"dproc_periods\": %d, "
                "\"queue_periods\": %d, \"profile\": %s, \"budget\": %s, \"result\": %s}\n",
                q(c.id).c_str(), t.period, t.hop_periods, t.dproc_periods, t.queue_periods,
                q(a.has("proc-ms") ? "fixed" : prof).c_str(), budget_json(b, why).c_str(),
                why.empty() ? r.json().c_str() : "null");
    return why.empty() ? 0 : 3;
}

// ---- live ------------------------------------------------------------------------------------------------------------------
#ifdef VLD_WITH_ALSA
std::string set_rt(int prio, int cpu, bool enable) {
    if (!enable) return "\"off\"";
    std::string s = "{";
    sched_param sp{};
    sp.sched_priority = prio;
    const int e1 = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    s += "\"sched_fifo\": " + std::string(e1 == 0 ? "true" : "false") + ", \"prio\": " + std::to_string(prio);
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        const int e2 = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        s += ", \"cpu\": " + std::to_string(cpu) + ", \"pinned\": " + (e2 == 0 ? "true" : "false");
    }
    return s + "}";
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::string s;
    std::getline(f, s);
    return s;
}

int run_live(const Args& a) {
    const std::string cid = a.need("contract");
    const Contract c = parse_contract_id(cid);
    const Timeline t = timeline_of(a, c);
    const Fir fir = load_fir(a.need("resampler"));
    Budget b;
    const auto why = check_timeline(t, c.hop, c.support, fir.pair_peak_ms, &b);
    if (!a.has("dproc")) throw std::runtime_error("--dproc (whole periods, from Gate 0a) is required in live mode");
    if (!why.empty()) {
        std::printf("{\"mode\": \"live\", \"refused\": true, \"budget\": %s}\n", budget_json(b, why).c_str());
        return 3;
    }
    if (!b.eligible() && !a.has("allow-ineligible")) {
        std::printf("{\"mode\": \"live\", \"refused\": true, \"reason\": \"upper estimate above 13.0 ms (Section 4); "
                    "--allow-ineligible runs it as a non-qualifying control\", \"budget\": %s}\n",
                    budget_json(b, why).c_str());
        return 3;
    }
    const bool rt = !a.has("no-rt");
    const int cpu = static_cast<int>(a.num("cpu", -1)), prio = static_cast<int>(a.num("prio", 80));
    const int mlock_ok = rt ? mlockall(MCL_CURRENT | MCL_FUTURE) : -1;
    const std::string rt_main = set_rt(prio, cpu, rt);
    flush_denormals(true);

    EngineConfig cfg = engine_config(a);
    Engine e(a.need("model"), cid, cfg);
    Engine48 e48(e, fir);
    const int P = t.period, n = t.hop_periods, N = t.hop_frames(), dq = static_cast<int>(t.delay_frames() / P);
    const int pc = static_cast<int>(a.num("prim-ch", 0)), rc = static_cast<int>(a.num("ref-ch", 1));
    std::vector<float> p48(N), r48(N), y48(N);
    std::vector<uint8_t> av48(N, 1);

    // warmup before capture: the model, the resamplers and every buffer run hot, then the stream starts fresh
    LatencyHist warm;
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.0f, 0.01f);
    for (int j = 0; j < static_cast<int>(a.num("warmup-hops", 500)); ++j) {
        for (int i = 0; i < N; ++i) { p48[i] = nd(g); r48[i] = nd(g); }
        const uint64_t t0 = now_ns();
        e48.process(p48.data(), r48.data(), av48.data(), y48.data());
        warm.add(now_ns() - t0);
    }
    e48.reset();

    AlsaConfig ac;
    ac.capture = a.get("capture", ac.capture);
    ac.playback = a.get("playback", ac.playback);
    ac.period = P;
    ac.buffer_periods = std::max(t.queue_periods + 2, static_cast<int>(a.num("buffer-periods", 4)));
    ac.allow_plugin = a.has("allow-plugin");
    AlsaDuplex dev(ac);
    std::string ctl_json = "[";
    for (size_t i = 0; i < a.ctl.size(); ++i) {
        const std::string& s = a.ctl[i];
        const size_t eq = s.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--ctl takes NAME=VALUE");
        const std::string v = alsa_set_control(alsa_card_of(ac.playback), s.substr(0, eq), s.substr(eq + 1));
        ctl_json += (i ? ", " : "") + std::string("{\"name\": ") + q(s.substr(0, eq)) + ", \"value\": " + q(v) + "}";
    }
    ctl_json += "]";

    PeriodRing ring(t.ring_periods(), P);
    const unsigned chi = ac.channels_in, cho = ac.channels_out;
    std::atomic<int64_t> w_next{0};
    std::atomic<bool> running{true}, paused{false}, ack{false}, dev_error{false};
    std::atomic<int64_t> late_periods{0}, stale_periods{0}, silence_periods{0}, write_errors{0};
    std::string rt_writer;
    std::atomic<bool> writer_ready{false};

    // period writer: keeps q periods queued in the device, each from the ring by its output index, else silence
    std::thread writer([&] {
        rt_writer = set_rt(prio + 1, cpu, rt);
        flush_denormals(true);
        writer_ready = true;
        std::vector<float> f(P);
        std::vector<int32_t> ob(static_cast<size_t>(P) * cho);
        int64_t stale = 0;
        while (running.load()) {
            if (paused.load()) { ack = true; usleep(100); continue; }
            ack = false;
            if (dev.wait_playback(10) < 0) { dev_error = true; usleep(100); continue; }
            long avail = dev.playback_avail();
            if (avail < 0) { dev_error = true; usleep(100); continue; }
            long occ = ac.period * ac.buffer_periods - avail;
            while (occ <= static_cast<long>(t.queue_periods - 1) * P && !paused.load()) {
                const int64_t w = w_next.load(), m = w - dq;
                const auto k = ring.take(m, f.data(), &stale);
                if (k == PeriodRing::LATE) ++late_periods;
                else if (k == PeriodRing::SILENCE_START) ++silence_periods;
                for (int i = 0; i < P; ++i) {
                    const float v = std::max(-1.0f, std::min(1.0f, std::isfinite(f[i]) ? f[i] : 0.0f));
                    const int32_t s = static_cast<int32_t>(std::lrint(static_cast<double>(v) * 2147483647.0));
                    for (unsigned ch = 0; ch < cho; ++ch) ob[i * cho + ch] = s;
                }
                if (dev.write_period(ob.data()) != P) { ++write_errors; dev_error = true; break; }
                w_next.store(w + 1);
                occ += P;
            }
            stale_periods.store(stale);
        }
    });
    while (!writer_ready.load()) usleep(100);

    LatencyHist h_total, h_wake, h_front, h_an, h_step, h_syn, h_rs;
    int64_t hops = 0, hop_index = 0, deadline_miss = 0, dropped = 0, xruns = 0, ring_full = 0;
    int64_t frames_read = 0, drift_min = INT64_MAX, drift_max = INT64_MIN, drift_first = 0, drift_last = 0, drift_n = 0;
    const double ns_per_frame = 1e9 / ac.rate;
    std::vector<int32_t> cb(static_cast<size_t>(P) * chi);

    auto restart = [&] {
        paused = true;
        while (!ack.load()) usleep(50);
        dev.stop();
        // drain the ring (the writer is paused, so this thread may consume): anything left is stale
        std::vector<float> tmp(P);
        int64_t st = 0;
        while (ring.occupancy() > 0) ring.take(INT64_MAX, tmp.data(), &st);
        e48.dec.reset(); e48.itp.reset();
        e.begin_recovery();
        hop_index = 0; frames_read = 0;
        w_next.store(t.queue_periods);
        dev.start(t.queue_periods);
        dev_error = false;
        paused = false;
        ++xruns;
    };

    const double duration = a.num("duration", 60);
    const int64_t max_hops = static_cast<int64_t>(duration * 48000 / N);
    w_next.store(t.queue_periods);
    dev.start(t.queue_periods);
    while (hops < max_hops) {
        bool bad = false;
        for (int i = 0; i < n && !bad; ++i) {
            const long rr = dev.read_period(cb.data());
            if (rr < 0) { bad = true; break; }
            for (int k = 0; k < P; ++k) {
                p48[i * P + k] = cb[k * chi + pc] / 2147483648.0f;
                r48[i * P + k] = cb[k * chi + rc] / 2147483648.0f;
            }
        }
        if (bad || dev_error.load()) { restart(); continue; }
        frames_read += N;
        uint64_t st_ns = 0;
        long st_av = 0;
        const uint64_t woke = now_ns();
        uint64_t t_complete = woke;
        if (dev.capture_status(&st_ns, &st_av) && st_ns) t_complete = st_ns - static_cast<uint64_t>(st_av * ns_per_frame);
        const uint64_t deadline = t_complete + static_cast<uint64_t>(t.dproc_periods * P * ns_per_frame);
        e48.process(p48.data(), r48.data(), av48.data(), y48.data());
        const uint64_t done = now_ns();
        h_total.add(done > t_complete ? done - t_complete : 0);
        h_wake.add(woke > t_complete ? woke - t_complete : 0);
        h_front.add(e.stage_ns.frontend); h_an.add(e.stage_ns.analysis); h_step.add(e.stage_ns.step);
        h_syn.add(e.stage_ns.synthesis); h_rs.add(e.stage_ns.resample);
        if (done > deadline) ++deadline_miss;
        const int64_t m0 = hop_index * n;
        if (m0 + dq < w_next.load()) {             // its first period's write time has passed: stale work
            ++dropped;
            e.begin_recovery();
        } else {
            for (int i = 0; i < n; ++i)
                if (!ring.push(m0 + i, &y48[i * P], done)) ++ring_full;
        }
        ++hop_index; ++hops;
        if (hops % 250 == 0) {                     // shared-clock check: capture minus playback position
            const long ca = dev.capture_avail(), pd = dev.playback_delay();
            if (ca >= 0 && pd >= 0) {
                const int64_t diff = (frames_read + ca) - (w_next.load() * P - pd);
                if (!drift_n) drift_first = diff;
                drift_last = diff; ++drift_n;
                drift_min = std::min(drift_min, diff); drift_max = std::max(drift_max, diff);
            }
        }
    }
    running = false;
    writer.join();
    dev.stop();

    const double secs = hops * static_cast<double>(N) / 48000.0;
    const double drift_ppm = drift_n > 1 && secs > 0 ? (drift_last - drift_first) / (secs * 48000.0) * 1e6 : 0.0;
    const bool shared = dev.same_card() && drift_n > 1 && (drift_max - drift_min) <= 2 * P;
    const bool qualifies = b.eligible() && shared && !deadline_miss && !dropped && !xruns && !late_periods.load() &&
                           !stale_periods.load() && !ring_full && !write_errors.load() && !e.recoveries &&
                           mlock_ok == 0 && rt_main.find("\"sched_fifo\": true") != std::string::npos;
    std::ostringstream o;
    o << "{\"mode\": \"live\", \"contract\": " << q(cid) << ", \"resampler\": " << q(fir.id)
      << ", \"resampler_sha256\": " << q(fir.sha256) << ", \"period\": " << P << ", \"hop_periods\": " << n
      << ", \"dproc_periods\": " << t.dproc_periods << ", \"queue_periods\": " << t.queue_periods
      << ", \"delay_frames\": " << t.delay_frames() << ", \"path_delay_ms_upper\": " << b.total_ms()
      << ", \"budget\": " << budget_json(b, why) << ", \"alsa\": " << dev.json() << ", \"controls\": " << ctl_json
      << ", \"kernel\": " << q(kernel()) << ", \"rt_kernel\": " << (read_file("/sys/kernel/realtime") == "1" ? "true" : "false")
      << ", \"governor\": " << q(read_file("/sys/devices/system/cpu/cpu" + std::to_string(std::max(cpu, 0)) + "/cpufreq/scaling_governor"))
      << ", \"mlockall\": " << (mlock_ok == 0 ? "true" : "false") << ", \"rt_main\": " << rt_main
      << ", \"rt_writer\": " << rt_writer << ", \"seconds\": " << secs << ", \"hops\": " << hops
      << ", \"deadline_miss\": " << deadline_miss << ", \"dropped_stale_hops\": " << dropped << ", \"xruns\": " << xruns
      << ", \"recoveries\": " << e.recoveries << ", \"late_periods\": " << late_periods.load()
      << ", \"stale_periods\": " << stale_periods.load() << ", \"ring_full\": " << ring_full
      << ", \"write_errors\": " << write_errors.load() << ", \"ring_max\": " << ring.max_occupancy
      << ", \"ring_capacity\": " << ring.capacity() << ", \"drift\": {\"frames_min\": " << (drift_n ? drift_min : 0)
      << ", \"frames_max\": " << (drift_n ? drift_max : 0) << ", \"ppm\": " << drift_ppm << ", \"shared_clock\": "
      << (shared ? "true" : "false") << "}, \"warmup_hop\": " << warm.json() << ", \"wake\": " << h_wake.json()
      << ", \"wake_plus_processing\": " << h_total.json() << ", \"stages\": {\"frontend\": " << h_front.json()
      << ", \"analysis\": " << h_an.json() << ", \"step\": " << h_step.json() << ", \"synthesis\": " << h_syn.json()
      << ", \"resample\": " << h_rs.json() << "}, \"qualifies\": " << (qualifies ? "true" : "false") << "}";
    const std::string rep = o.str();
    std::printf("%s\n", rep.c_str());
    if (a.has("report")) std::ofstream(a.get("report")) << rep << "\n";
    return qualifies ? 0 : 4;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse(argc, argv);
        if (a.mode == "info") return run_info(a);
        if (a.mode == "wav") return run_wav(a);
        if (a.mode == "simulate") return run_simulate(a);
#ifdef VLD_WITH_ALSA
        if (a.mode == "live") return run_live(a);
#else
        if (a.mode == "live") throw std::runtime_error("built without ALSA (VLD_WITH_ALSA=OFF): live mode unavailable");
#endif
        throw std::runtime_error("unknown mode " + a.mode);
    } catch (const std::exception& ex) {
        std::printf("{\"error\": %s}\n", q(ex.what()).c_str());
        return 1;
    }
}
