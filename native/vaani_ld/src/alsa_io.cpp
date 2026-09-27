// See include/vaani_ld/alsa_io.hpp.
#include "vaani_ld/alsa_io.hpp"

#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace vld {

static void ck(int err, const std::string& what) {
    if (err < 0) throw std::runtime_error("ALSA " + what + ": " + snd_strerror(err));
}

static std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

std::string AlsaSide::json() const {
    std::ostringstream o;
    o << "{\"name\": \"" << esc(name) << "\", \"type\": \"" << esc(type) << "\", \"card\": " << card
      << ", \"card_driver\": \"" << esc(card_driver) << "\", \"card_name\": \"" << esc(card_name)
      << "\", \"pcm_id\": \"" << esc(pcm_id) << "\", \"rate\": " << rate << ", \"channels\": " << channels
      << ", \"format\": \"" << format << "\", \"period\": " << period << ", \"period_min\": " << period_min
      << ", \"buffer\": " << buffer << "}";
    return o.str();
}

static void configure(snd_pcm_t* pcm, const AlsaConfig& cfg, unsigned channels, bool capture, AlsaSide& side) {
    const std::string dir = capture ? "capture" : "playback";
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    ck(snd_pcm_hw_params_any(pcm, hw), dir + " hw_params_any");
    ck(snd_pcm_hw_params_set_rate_resample(pcm, hw, 0), dir + " disable resampling");
    ck(snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED), dir + " interleaved access");
    ck(snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S32_LE), dir + " S32_LE");
    ck(snd_pcm_hw_params_set_channels(pcm, hw, channels), dir + " channels " + std::to_string(channels));
    ck(snd_pcm_hw_params_set_rate(pcm, hw, cfg.rate, 0), dir + " rate " + std::to_string(cfg.rate) + " exactly");
    snd_pcm_uframes_t pmin = 0;
    int sub = 0;
    snd_pcm_hw_params_get_period_size_min(hw, &pmin, &sub);
    side.period_min = static_cast<long>(pmin);
    ck(snd_pcm_hw_params_set_period_size(pcm, hw, cfg.period, 0),
       dir + " period " + std::to_string(cfg.period) + " frames exactly (driver minimum " + std::to_string(pmin) + ")");
    ck(snd_pcm_hw_params_set_buffer_size(pcm, hw, static_cast<snd_pcm_uframes_t>(cfg.period) * cfg.buffer_periods),
       dir + " buffer " + std::to_string(cfg.buffer_periods) + " periods");
    ck(snd_pcm_hw_params(pcm, hw), dir + " hw_params");
    // read back and verify: requesting a setting does not prove it
    unsigned rate = 0, ch = 0;
    int d = 0;
    snd_pcm_uframes_t per = 0, buf = 0;
    snd_pcm_format_t fmt;
    snd_pcm_hw_params_get_rate(hw, &rate, &d);
    snd_pcm_hw_params_get_channels(hw, &ch);
    snd_pcm_hw_params_get_period_size(hw, &per, &d);
    snd_pcm_hw_params_get_buffer_size(hw, &buf);
    snd_pcm_hw_params_get_format(hw, &fmt);
    side.rate = rate; side.channels = ch; side.period = static_cast<long>(per); side.buffer = static_cast<long>(buf);
    side.format = snd_pcm_format_name(fmt);
    if (rate != cfg.rate || ch != channels || per != static_cast<snd_pcm_uframes_t>(cfg.period) ||
        fmt != SND_PCM_FORMAT_S32_LE)
        throw std::runtime_error(dir + ": negotiated " + std::to_string(rate) + " Hz, " + std::to_string(ch) + " ch, " +
                                 std::to_string(per) + "-frame periods, " + side.format + " differs from the request");

    snd_pcm_sw_params_t* sw;
    snd_pcm_sw_params_alloca(&sw);
    ck(snd_pcm_sw_params_current(pcm, sw), dir + " sw_params_current");
    ck(snd_pcm_sw_params_set_avail_min(pcm, sw, cfg.period), dir + " avail_min");
    // started explicitly (and together, through the link), never by a fill threshold
    ck(snd_pcm_sw_params_set_start_threshold(pcm, sw, capture ? 1 : buf * 2), dir + " start threshold");
    ck(snd_pcm_sw_params_set_tstamp_mode(pcm, sw, SND_PCM_TSTAMP_ENABLE), dir + " timestamps");
    ck(snd_pcm_sw_params_set_tstamp_type(pcm, sw, SND_PCM_TSTAMP_TYPE_MONOTONIC), dir + " monotonic timestamps");
    ck(snd_pcm_sw_params(pcm, sw), dir + " sw_params");

    side.name = snd_pcm_name(pcm);
    side.type = snd_pcm_type_name(snd_pcm_type(pcm));
    snd_pcm_info_t* info;
    snd_pcm_info_alloca(&info);
    if (snd_pcm_info(pcm, info) == 0) {
        side.card = snd_pcm_info_get_card(info);
        side.pcm_id = snd_pcm_info_get_id(info);
        snd_ctl_t* ctl;
        if (side.card >= 0 && snd_ctl_open(&ctl, ("hw:" + std::to_string(side.card)).c_str(), 0) == 0) {
            snd_ctl_card_info_t* ci;
            snd_ctl_card_info_alloca(&ci);
            if (snd_ctl_card_info(ctl, ci) == 0) { side.card_driver = snd_ctl_card_info_get_driver(ci); side.card_name = snd_ctl_card_info_get_name(ci); }
            snd_ctl_close(ctl);
        }
    }
    if (snd_pcm_type(pcm) != SND_PCM_TYPE_HW && !cfg.allow_plugin)
        throw std::runtime_error(dir + " PCM " + side.name + " is a " + side.type +
                                 " plugin (possible internal conversion); open a hw: device");
}

AlsaDuplex::AlsaDuplex(const AlsaConfig& c) : cfg(c) {
    ck(snd_pcm_open(&cap, cfg.capture.c_str(), SND_PCM_STREAM_CAPTURE, 0), "open capture " + cfg.capture);
    ck(snd_pcm_open(&play, cfg.playback.c_str(), SND_PCM_STREAM_PLAYBACK, 0), "open playback " + cfg.playback);
    configure(cap, cfg, cfg.channels_in, true, cap_info);
    configure(play, cfg, cfg.channels_out, false, play_info);
    const int err = snd_pcm_link(cap, play);
    if (err < 0)
        throw std::runtime_error(std::string("snd_pcm_link: ") + snd_strerror(err) +
                                 " (capture and playback must start on the same frame)");
    linked = true;
    silence.assign(static_cast<size_t>(cfg.period) * cfg.channels_out, 0);
}

AlsaDuplex::~AlsaDuplex() {
    if (cap) { if (linked) snd_pcm_unlink(cap); snd_pcm_drop(cap); snd_pcm_close(cap); }
    if (play) { snd_pcm_drop(play); snd_pcm_close(play); }
}

void AlsaDuplex::start(int prefill) {
    snd_pcm_drop(cap);
    ck(snd_pcm_prepare(cap), "prepare capture");
    if (snd_pcm_state(play) != SND_PCM_STATE_PREPARED) ck(snd_pcm_prepare(play), "prepare playback");
    for (int i = 0; i < prefill; ++i)
        if (snd_pcm_writei(play, silence.data(), cfg.period) != cfg.period)
            throw std::runtime_error("playback prefill failed");
    ck(snd_pcm_start(cap), "linked start");
}

void AlsaDuplex::stop() { snd_pcm_drop(cap); snd_pcm_drop(play); }

long AlsaDuplex::read_period(int32_t* buf) {
    long got = 0;
    while (got < cfg.period) {
        const snd_pcm_sframes_t r = snd_pcm_readi(cap, buf + got * cfg.channels_in, cfg.period - got);
        if (r < 0) return r;
        got += r;
    }
    return got;
}

long AlsaDuplex::write_period(const int32_t* buf) {
    const snd_pcm_sframes_t r = snd_pcm_writei(play, buf, cfg.period);
    return r;
}

long AlsaDuplex::capture_avail() { return snd_pcm_avail(cap); }
long AlsaDuplex::playback_avail() { return snd_pcm_avail_update(play); }
long AlsaDuplex::playback_delay() {
    snd_pcm_sframes_t d = 0;
    const int e = snd_pcm_delay(play, &d);
    return e < 0 ? e : d;
}
int AlsaDuplex::wait_playback(int timeout_ms) { return snd_pcm_wait(play, timeout_ms); }

uint64_t AlsaDuplex::capture_trigger_ns() {
    snd_pcm_status_t* st;
    snd_pcm_status_alloca(&st);
    if (snd_pcm_status(cap, st) < 0) return 0;
    snd_htimestamp_t ts;
    snd_pcm_status_get_trigger_htstamp(st, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

bool AlsaDuplex::capture_status(uint64_t* ns, long* avail) {
    snd_pcm_status_t* st;
    snd_pcm_status_alloca(&st);
    if (snd_pcm_status(cap, st) < 0) return false;
    snd_htimestamp_t ts;
    snd_pcm_status_get_htstamp(st, &ts);
    *ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    *avail = static_cast<long>(snd_pcm_status_get_avail(st));
    return true;
}

std::string AlsaDuplex::json() const {
    return "{\"capture\": " + cap_info.json() + ", \"playback\": " + play_info.json() +
           ", \"linked\": " + (linked ? "true" : "false") + ", \"same_card\": " + (same_card() ? "true" : "false") + "}";
}

// ---- controls -------------------------------------------------------------------------------------------------------
std::string alsa_card_of(const std::string& pcm) {
    const size_t c = pcm.find(':');
    if (c == std::string::npos) return "default";
    std::string card = pcm.substr(c + 1);
    card = card.substr(0, card.find(','));
    return "hw:" + card;
}

static std::string read_control(snd_ctl_t* ctl, snd_ctl_elem_id_t* id) {
    snd_ctl_elem_info_t* info;
    snd_ctl_elem_value_t* val;
    snd_ctl_elem_info_alloca(&info);
    snd_ctl_elem_value_alloca(&val);
    snd_ctl_elem_info_set_id(info, id);
    ck(snd_ctl_elem_info(ctl, info), "control info");
    snd_ctl_elem_value_set_id(val, id);
    ck(snd_ctl_elem_read(ctl, val), "control read");
    if (snd_ctl_elem_info_get_type(info) == SND_CTL_ELEM_TYPE_ENUMERATED) {
        const unsigned item = snd_ctl_elem_value_get_enumerated(val, 0);
        snd_ctl_elem_info_set_item(info, item);
        ck(snd_ctl_elem_info(ctl, info), "control item");
        return snd_ctl_elem_info_get_item_name(info);
    }
    return std::to_string(snd_ctl_elem_value_get_integer(val, 0));
}

static snd_ctl_t* open_ctl(const std::string& card) {
    snd_ctl_t* ctl;
    ck(snd_ctl_open(&ctl, card.c_str(), 0), "open control device " + card);
    return ctl;
}

std::string alsa_get_control(const std::string& card, const std::string& name) {
    snd_ctl_t* ctl = open_ctl(card);
    snd_ctl_elem_id_t* id;
    snd_ctl_elem_id_alloca(&id);
    snd_ctl_elem_id_set_interface(id, SND_CTL_ELEM_IFACE_MIXER);
    snd_ctl_elem_id_set_name(id, name.c_str());
    try {
        std::string v = read_control(ctl, id);
        snd_ctl_close(ctl);
        return v;
    } catch (...) { snd_ctl_close(ctl); throw; }
}

std::string alsa_set_control(const std::string& card, const std::string& name, const std::string& value) {
    snd_ctl_t* ctl = open_ctl(card);
    snd_ctl_elem_id_t* id;
    snd_ctl_elem_info_t* info;
    snd_ctl_elem_value_t* val;
    snd_ctl_elem_id_alloca(&id);
    snd_ctl_elem_info_alloca(&info);
    snd_ctl_elem_value_alloca(&val);
    snd_ctl_elem_id_set_interface(id, SND_CTL_ELEM_IFACE_MIXER);
    snd_ctl_elem_id_set_name(id, name.c_str());
    try {
        snd_ctl_elem_info_set_id(info, id);
        ck(snd_ctl_elem_info(ctl, info), "control \"" + name + "\" on " + card);
        snd_ctl_elem_value_set_id(val, id);
        ck(snd_ctl_ascii_value_parse(ctl, val, info, value.c_str()), "value \"" + value + "\" for \"" + name + "\"");
        ck(snd_ctl_elem_write(ctl, val), "write control \"" + name + "\"");
        std::string v = read_control(ctl, id);
        snd_ctl_close(ctl);
        return v;
    } catch (...) { snd_ctl_close(ctl); throw; }
}

}  // namespace vld
