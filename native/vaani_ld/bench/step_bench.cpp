// Gate 0a step timing on the target (Task 0/7): the whole native hop and the ONNX Runtime step alone, paced at the
// hop rate with clock_nanosleep, after warmup, with FZ on or off, over random, silent and low-level inputs; and the
// inference library's allocation behaviour after warmup (operator new calls and mallinfo2 arena growth), measured
// separately from the host-side DSP, which allocates nothing.
//   vld_step_bench --model M.onnx --contract ID [--hops 20000] [--warmup 500] [--input random|silent|lowlevel]
//                  [--fz on|off] [--paced 1] [--resampler R.json] [--cpu N] [--prio 80]
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/utsname.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "vaani_ld/engine.hpp"
#include "vaani_ld/stats.hpp"

static std::atomic<uint64_t> g_news{0};
void* operator new(size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

using namespace vld;

int main(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[std::string(argv[i]).substr(2)] = argv[i + 1];
    auto get = [&](const char* k, const char* d) { return kv.count(k) ? kv[k] : std::string(d); };
    try {
        if (!kv.count("model") || !kv.count("contract")) throw std::runtime_error("--model and --contract are required");
        const int hops = std::stoi(get("hops", "20000")), warm = std::stoi(get("warmup", "500"));
        const std::string input = get("input", "random");
        const bool fz = get("fz", "on") == "on", paced = get("paced", "1") == "1";
        const int cpu = std::stoi(get("cpu", "-1")), prio = std::stoi(get("prio", "0"));
        std::string rt = "{}";
        if (prio > 0) {
            sched_param sp{}; sp.sched_priority = prio;
            const bool ok = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
            rt = std::string("{\"sched_fifo\": ") + (ok ? "true" : "false") + ", \"mlockall\": " +
                 (mlockall(MCL_CURRENT | MCL_FUTURE) == 0 ? "true" : "false") + "}";
        }
        if (cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
        flush_denormals(fz);
        EngineConfig cfg;
        cfg.timing = true;
        cfg.denormal_as_zero = fz;
        Engine e(kv["model"], kv["contract"], cfg);
        std::unique_ptr<Fir> fir;
        std::unique_ptr<Engine48> e48;
        if (kv.count("resampler")) { fir.reset(new Fir(load_fir(kv["resampler"]))); e48.reset(new Engine48(e, *fir)); }
        const int H = e.contract().hop, rate = e48 ? 3 : 1, N = H * rate;
        std::vector<float> p(N), r(N), y(N);
        std::vector<uint8_t> av(N, 1);
        std::mt19937 g(0);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const float scale = input == "random" ? 0.1f : input == "lowlevel" ? 1e-6f : 0.0f;
        if (input != "random" && input != "silent" && input != "lowlevel") throw std::runtime_error("unknown --input");
        auto fill = [&] { for (int i = 0; i < N; ++i) { p[i] = scale * nd(g); r[i] = scale * nd(g); } };
        auto hop = [&] {
            if (e48) e48->process(p.data(), r.data(), av.data(), y.data());
            else e.process_hop(p.data(), r.data(), av.data(), y.data());
        };
        for (int j = 0; j < warm; ++j) { fill(); hop(); }
        LatencyHist whole, step;
        const uint64_t period_ns = static_cast<uint64_t>(H) * 1000000000ULL / 16000;
        timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);
        const struct mallinfo2 m0 = mallinfo2();
        uint64_t news = 0;
        for (int j = 0; j < hops; ++j) {
            fill();                                        // the generator is outside the counted span
            const uint64_t n0 = g_news.load(), t0 = now_ns();
            hop();
            const uint64_t t1 = now_ns();
            news += g_news.load() - n0;
            whole.add(t1 - t0);
            step.add(e.stage_ns.step);
            if (paced) {
                next.tv_nsec += static_cast<long>(period_ns);
                while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; ++next.tv_sec; }
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
            }
        }
        const struct mallinfo2 m1 = mallinfo2();
        utsname u{};
        uname(&u);
        std::printf("{\"contract\": \"%s\", \"state_floats\": %d, \"input\": \"%s\", \"fz\": %s, \"paced\": %s, "
                    "\"resampler\": %s, \"hops\": %d, \"warmup\": %d, \"rt\": %s, \"machine\": \"%s %s\", "
                    "\"whole_hop\": %s, \"step\": %s, \"allocations_after_warmup\": %llu, "
                    "\"arena_growth_bytes\": %lld, \"hop_budget_ms\": %.3f}\n",
                    e.contract().id.c_str(), e.state_floats(), input.c_str(), fz ? "true" : "false",
                    paced ? "true" : "false", fir ? ("\"" + fir->id + "\"").c_str() : "null", hops, warm, rt.c_str(),
                    u.machine, u.release, whole.json().c_str(), step.json().c_str(),
                    static_cast<unsigned long long>(news),
                    static_cast<long long>(m1.arena + m1.hblkhd) - static_cast<long long>(m0.arena + m0.hblkhd),
                    H / 16.0);
    } catch (const std::exception& ex) {
        std::printf("{\"error\": \"%s\"}\n", ex.what());
        return 1;
    }
    return 0;
}
