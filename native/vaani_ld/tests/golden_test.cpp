// Golden-vector parity (Task 7): the native DSP and engine against the Task 6 vectors.
//   vld_golden_test VECTORS_NPY_DIR MODELS_DIR RESAMPLER_JSON_DIR [TOL]
// VECTORS_NPY_DIR holds <contract>/<case>/<key>.npy and resampler/<id>/<key>.npy (the golden .npz files, extracted
// by tests/test_low_delay_native.py); MODELS_DIR holds <contract>/model.onnx; RESAMPLER_JSON_DIR the coefficient
// files. Prints one JSON object per check and a summary; exits 1 when any check exceeds TOL (default 1e-5).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <string>
#include <vector>

#include "npy.hpp"
#include "vaani_ld/engine.hpp"

using namespace vld;

static double TOL = 1e-5;
static int failures = 0, checks = 0;

static double max_abs(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool fa = std::isfinite(a[i]), fb = std::isfinite(b[i]);
        if (fa != fb) return INFINITY;
        if (fa) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return m;
}

static void report(const std::string& where, const std::string& what, double err, double tol = TOL) {
    ++checks;
    const bool ok = err <= tol;
    if (!ok) ++failures;
    std::printf("{\"where\": \"%s\", \"check\": \"%s\", \"max_abs\": %.3g, \"tol\": %.3g, \"pass\": %s}\n",
                where.c_str(), what.c_str(), err, tol, ok ? "true" : "false");
}

static std::vector<std::string> subdirs(const std::string& d) {
    std::vector<std::string> out;
    DIR* dir = opendir(d.c_str());
    if (!dir) return out;
    while (dirent* e = readdir(dir)) {
        std::string n = e->d_name;
        if (n != "." && n != ".." && e->d_type == DT_DIR) out.push_back(n);
    }
    closedir(dir);
    std::sort(out.begin(), out.end());
    return out;
}

static bool exists(const std::string& p) { return std::ifstream(p).good(); }

static void resampler_case(const std::string& dir, const std::string& json) {
    const Fir fir = load_fir(json);
    const auto h = npy::load(dir + "/h.npy").f64();
    double herr = 0;
    for (size_t i = 0; i < h.size(); ++i) herr = std::max(herr, std::fabs(h[i] - fir.h[i]));
    report(fir.id, "coefficients", h.size() == fir.h.size() ? herr : INFINITY, 0.0);
    const auto x16 = npy::load(dir + "/x16.npy").f32(), x48 = npy::load(dir + "/x48.npy").f32();
    const auto blocks = npy::load(dir + "/blocks.npy").i64();
    const auto yi = npy::load(dir + "/interpolated.npy").f32(), yd = npy::load(dir + "/decimated.npy").f32();
    const size_t n16 = x16.size() / 2, n48 = x48.size() / 2;
    Interpolate3 itp(2, fir.h);
    Decimate3 dec(2, fir.h, 3 * 1068);
    std::vector<float> oi(2 * n48), od(2 * n16), bi, bo;
    size_t a = 0;
    for (int64_t b : blocks) {
        bi.assign(2 * b, 0); bo.assign(2 * 3 * b, 0);
        for (int c = 0; c < 2; ++c) for (int64_t i = 0; i < b; ++i) bi[c * b + i] = x16[c * n16 + a + i];
        itp.process(bi.data(), static_cast<int>(b), bo.data());
        for (int c = 0; c < 2; ++c) for (int64_t i = 0; i < 3 * b; ++i) oi[c * n48 + 3 * a + i] = bo[c * 3 * b + i];
        bi.assign(2 * 3 * b, 0); bo.assign(2 * b, 0);
        for (int c = 0; c < 2; ++c) for (int64_t i = 0; i < 3 * b; ++i) bi[c * 3 * b + i] = x48[c * n48 + 3 * a + i];
        dec.process(bi.data(), static_cast<int>(3 * b), bo.data());
        for (int c = 0; c < 2; ++c) for (int64_t i = 0; i < b; ++i) od[c * n16 + a + i] = bo[c * b + i];
        a += b;
    }
    report(fir.id, "interpolate3", max_abs(oi.data(), yi.data(), oi.size()));
    report(fir.id, "decimate3", max_abs(od.data(), yd.data(), od.size()));
}

static void engine_case(const std::string& cdir, const std::string& model, const std::string& cid,
                        const std::string& name) {
    const std::string d = cdir + "/" + name, where = cid + "/" + name;
    EngineConfig cfg;
    Engine e(model, cid, cfg);
    const Contract& c = e.contract();
    const int H = c.hop, B = c.bins(), S = e.state_floats();
    const auto fin = npy::load(d + "/frontend_in.npy");
    const auto x = fin.f32();
    const auto av = npy::load(d + "/available.npy").u8();
    const auto syn = npy::load(d + "/synthesis.npy").f32();
    const auto fvalid = npy::load(d + "/frame_valid.npy").f32();
    const auto disc = npy::load(d + "/discontinuity.npy").u8();
    const auto fstate = npy::load(d + "/final_state.npy").f32();
    const size_t hops = fin.shape[0];
    const bool stage = exists(d + "/stage_hops.npy");
    std::vector<int64_t> sh, sth;
    std::vector<float> fo, an, so, st, li, lo;
    if (stage) {
        sh = npy::load(d + "/stage_hops.npy").i64(); sth = npy::load(d + "/state_hops.npy").i64();
        fo = npy::load(d + "/frontend_out.npy").f32(); an = npy::load(d + "/analysis.npy").f32();
        so = npy::load(d + "/step_out.npy").f32(); st = npy::load(d + "/state.npy").f32();
        li = npy::load(d + "/limiter_in.npy").f32(); lo = npy::load(d + "/limiter_out.npy").f32();
    }
    Trace tr;
    e.trace = &tr;
    std::vector<float> y(hops * H);
    double e_fo = 0, e_an = 0, e_so = 0, e_st = 0, e_valid = 0;
    int disc_mismatch = 0;
    size_t si = 0, ti = 0;
    for (size_t j = 0; j < hops; ++j) {
        if (j + 1 == hops) e.flush_hop(y.data() + j * H);
        else e.process_hop(&x[j * 2 * H], &x[j * 2 * H + H], &av[j * H], y.data() + j * H);
        e_valid = std::max(e_valid, std::fabs(static_cast<double>(tr.frame_valid) - fvalid[j]));
        disc_mismatch += (tr.discontinuity ? 1 : 0) != disc[j];
        if (stage && si < sh.size() && static_cast<size_t>(sh[si]) == j) {
            e_fo = std::max(e_fo, max_abs(tr.frontend_out.data(), &fo[si * 2 * H], 2 * H));
            std::vector<float> a4(B * 4);
            for (int b = 0; b < B; ++b) {
                a4[4 * b] = tr.p[b].real(); a4[4 * b + 1] = tr.p[b].imag();
                a4[4 * b + 2] = tr.r[b].real(); a4[4 * b + 3] = tr.r[b].imag();
            }
            e_an = std::max(e_an, max_abs(a4.data(), &an[si * B * 4], B * 4));
            e_so = std::max(e_so, max_abs(tr.step_out.data(), &so[si * B * 2], B * 2));
            ++si;
        }
        if (stage && ti < sth.size() && static_cast<size_t>(sth[ti]) == j) {
            e_st = std::max(e_st, max_abs(e.neural_state().data(), &st[ti * S], S));
            ++ti;
        }
    }
    report(where, "synthesis", max_abs(y.data(), syn.data(), y.size()));
    report(where, "frame_valid", e_valid, 0.0);
    report(where, "discontinuity", disc_mismatch, 0.0);
    report(where, "final_state", max_abs(e.neural_state().data(), fstate.data(), S));
    if (stage) {
        report(where, "frontend_out", e_fo);
        report(where, "analysis", e_an);
        report(where, "step_out", e_so);
        report(where, "state", e_st);
        // the limiter alone over the contiguous startup hops (the frontend's limiter persists across hops)
        Limiter lim(c.limiter_sub);
        double e_li = 0;
        std::vector<float> p(H), r(H);
        for (size_t k = 0; k < sh.size() && static_cast<size_t>(sh[k]) == k && k + 1 < hops; ++k) {
            std::copy(&li[k * 2 * H], &li[k * 2 * H + H], p.begin());
            std::copy(&li[k * 2 * H + H], &li[k * 2 * H + 2 * H], r.begin());
            lim.process(p.data(), r.data(), H);
            e_li = std::max(e_li, std::max(max_abs(p.data(), &lo[k * 2 * H], H), max_abs(r.data(), &lo[k * 2 * H + H], H)));
        }
        report(where, "limiter", e_li);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s VECTORS_NPY_DIR MODELS_DIR RESAMPLER_JSON_DIR [TOL]\n", argv[0]);
        return 2;
    }
    const std::string vec = argv[1], models = argv[2], rjson = argv[3];
    if (argc > 4) TOL = std::atof(argv[4]);
    flush_denormals(false);   // the reference runs without FTZ; the golden low_level case must match bit-for-range
    try {
        for (const auto& id : subdirs(vec + "/resampler")) resampler_case(vec + "/resampler/" + id, rjson + "/" + id + ".json");
        for (const auto& cid : subdirs(vec)) {
            if (cid == "resampler") continue;
            for (const auto& name : subdirs(vec + "/" + cid))
                engine_case(vec + "/" + cid, models + "/" + cid + "/model.onnx", cid, name);
        }
    } catch (const std::exception& ex) {
        std::printf("{\"error\": \"%s\"}\n", ex.what());
        return 1;
    }
    std::printf("{\"summary\": true, \"checks\": %d, \"failures\": %d}\n", checks, failures);
    return failures || !checks ? 1 : 0;
}
