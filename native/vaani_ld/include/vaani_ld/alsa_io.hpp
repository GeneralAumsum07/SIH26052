// Direct ALSA duplex I/O for live mode and the Gate 0a period test (never arecord/aplay pipes).
// Capture and playback are opened on hw devices as interleaved S32_LE at exactly 48 kHz with an exact period
// (48 frames = 1 ms, or the 96-frame fallback), resampling disabled, linked with snd_pcm_link so both start on the
// same frame. Negotiation is inspected and anything other than the request is refused, as is a plugin PCM (internal
// conversion) unless explicitly allowed for a diagnostic run.
#pragma once
#include <alsa/asoundlib.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vld {

struct AlsaConfig {
    std::string capture = "hw:0,0", playback = "hw:0,0";
    unsigned rate = 48000;
    int period = 48;                // frames
    int buffer_periods = 4;         // device buffer (the writer keeps only q periods queued)
    unsigned channels_in = 2, channels_out = 2;
    bool allow_plugin = false;      // diagnostic only: a plugin PCM may convert internally
};

struct AlsaSide {
    std::string name, type, card_driver, card_name, pcm_id;
    int card = -1;
    unsigned rate = 0, channels = 0;
    long period = 0, period_min = 0, buffer = 0;
    std::string format;
    std::string json() const;
};

class AlsaDuplex {
public:
    explicit AlsaDuplex(const AlsaConfig& cfg);   // opens, negotiates and verifies; throws with the reason
    ~AlsaDuplex();
    AlsaDuplex(const AlsaDuplex&) = delete;
    AlsaDuplex& operator=(const AlsaDuplex&) = delete;

    // prepare both, prefill `prefill_periods` periods of silence, start (linked: one start for both)
    void start(int prefill_periods);
    void stop();
    // one period each; return frames or a negative errno (-EPIPE xrun, -ESTRPIPE suspend)
    long read_period(int32_t* interleaved);
    long write_period(const int32_t* interleaved);
    long capture_avail();
    long playback_delay();      // frames queued ahead of the DAC (snd_pcm_delay)
    long playback_avail();
    int wait_playback(int timeout_ms);
    uint64_t capture_trigger_ns();   // monotonic trigger timestamp of the linked start
    // monotonic time of the latest capture status, and the frames captured but not yet read at that time
    bool capture_status(uint64_t* ns, long* avail);

    const AlsaConfig& config() const { return cfg; }
    AlsaSide cap_info, play_info;
    bool linked = false;
    bool same_card() const { return cap_info.card >= 0 && cap_info.card == play_info.card; }
    std::string json() const;

    snd_pcm_t* cap = nullptr;
    snd_pcm_t* play = nullptr;
private:
    AlsaConfig cfg;
    std::vector<int32_t> silence;
};

// Set a mixer control by name, e.g. "DSP Program=Low latency IIR with de-emphasis" (the pcm512x interpolation
// filter). Returns the value read back; throws when the card or control is absent.
std::string alsa_set_control(const std::string& card, const std::string& name, const std::string& value);
std::string alsa_get_control(const std::string& card, const std::string& name);
// "hw:0,0" -> "hw:0" (the control device of a PCM's card)
std::string alsa_card_of(const std::string& pcm);

}  // namespace vld
