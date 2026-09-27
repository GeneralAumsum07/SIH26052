// Gate 0a period test: ALSA duplex identity passthrough on linked hw PCMs at 48 kHz with the requested period
// (48 frames = 1 ms by default), under SCHED_FIFO and mlockall. Reports the negotiated settings, the driver's minimum
// period, xruns, the worst wake-to-write time and the kernel, as one JSON object.
//   vld_period_test [--capture hw:0,0] [--playback hw:0,0] [--period 48] [--seconds 600] [--queue 2]
//                   [--cpu N] [--prio 80] [--ctl "NAME=VALUE"]
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/utsname.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "vaani_ld/alsa_io.hpp"
#include "vaani_ld/engine.hpp"
#include "vaani_ld/stats.hpp"

using namespace vld;

int main(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[std::string(argv[i]).substr(2)] = argv[i + 1];
    auto get = [&](const char* k, const char* d) { return kv.count(k) ? kv[k] : std::string(d); };
    try {
        AlsaConfig ac;
        ac.capture = get("capture", "hw:0,0");
        ac.playback = get("playback", "hw:0,0");
        ac.period = std::stoi(get("period", "48"));
        const int queue = std::stoi(get("queue", "2"));
        ac.buffer_periods = std::max(queue + 2, 4);
        const double seconds = std::stod(get("seconds", "600"));
        const int prio = std::stoi(get("prio", "80")), cpu = std::stoi(get("cpu", "-1"));
        sched_param sp{}; sp.sched_priority = prio;
        const bool fifo = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
        const bool locked = mlockall(MCL_CURRENT | MCL_FUTURE) == 0;
        if (cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
        AlsaDuplex dev(ac);
        std::string ctl = "null";
        if (kv.count("ctl")) {
            const std::string s = kv["ctl"];
            const size_t eq = s.find('=');
            ctl = "\"" + alsa_set_control(alsa_card_of(ac.playback), s.substr(0, eq), s.substr(eq + 1)) + "\"";
        }
        std::vector<int32_t> in(static_cast<size_t>(ac.period) * ac.channels_in), out(static_cast<size_t>(ac.period) * ac.channels_out);
        LatencyHist wake;
        long xruns = 0, periods = 0;
        const long total = static_cast<long>(seconds * ac.rate / ac.period);
        dev.start(queue);
        const double ns_per_frame = 1e9 / ac.rate;
        while (periods < total) {
            long r = dev.read_period(in.data());
            if (r >= 0) {
                uint64_t ts = 0; long av = 0;
                const uint64_t now = now_ns();
                if (dev.capture_status(&ts, &av) && ts) {
                    const uint64_t done = ts - static_cast<uint64_t>(av * ns_per_frame);
                    wake.add(now > done ? now - done : 0);
                }
                for (int i = 0; i < ac.period; ++i)
                    for (unsigned c = 0; c < ac.channels_out; ++c) out[i * ac.channels_out + c] = in[i * ac.channels_in + (c % ac.channels_in)];
                r = dev.write_period(out.data());
            }
            if (r < 0) { ++xruns; dev.stop(); dev.start(queue); continue; }
            ++periods;
        }
        dev.stop();
        utsname u{};
        uname(&u);
        std::printf("{\"alsa\": %s, \"queue_periods\": %d, \"seconds\": %.1f, \"periods\": %ld, \"xruns\": %ld, "
                    "\"wake\": %s, \"sched_fifo\": %s, \"mlockall\": %s, \"control\": %s, \"kernel\": \"%s %s %s\", "
                    "\"pass\": %s}\n",
                    dev.json().c_str(), queue, seconds, periods, xruns, wake.json().c_str(), fifo ? "true" : "false",
                    locked ? "true" : "false", ctl.c_str(), u.release, u.version, u.machine,
                    xruns == 0 && dev.same_card() ? "true" : "false");
        return xruns == 0 ? 0 : 4;
    } catch (const std::exception& ex) {
        std::printf("{\"error\": \"%s\"}\n", ex.what());
        return 1;
    }
}
