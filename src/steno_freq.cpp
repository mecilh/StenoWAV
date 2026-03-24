#include "steno.h"
#include "steno_freq.h"
#include "fft.h"
#include <cstdint>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <complex>

// --- Shared PRNG/utility (duplicated from steno.cpp to keep translation units independent) ---

static uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

class Xoshiro256ss {
    uint64_t s[4];
    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
public:
    void seed_from(uint64_t seed) {
        uint64_t sm = seed;
        s[0] = splitmix64(sm);
        s[1] = splitmix64(sm);
        s[2] = splitmix64(sm);
        s[3] = splitmix64(sm);
    }
    uint64_t next() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }
    uint32_t next_bounded(uint32_t bound) {
        return static_cast<uint32_t>(next() % bound);
    }
};

static uint64_t derive_seed(uint64_t nonce, const Wav::FmtChunk& fmt) {
    uint64_t mixed = nonce;
    mixed ^= static_cast<uint64_t>(fmt.AudioFormat) << 0;
    mixed ^= static_cast<uint64_t>(fmt.NbrChannel) << 16;
    mixed ^= static_cast<uint64_t>(fmt.Frequency) << 32;
    mixed ^= static_cast<uint64_t>(fmt.BitsPerSample) << 48;
    return splitmix64(mixed);
}

static std::string hex_encode(uint64_t val) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << val;
    return oss.str();
}

static uint64_t hex_decode(const std::string& hex) {
    uint64_t val = 0;
    std::istringstream iss(hex);
    iss >> std::hex >> val;
    return val;
}

// --- QIM parameters ---
// A single-bin magnitude change of delta produces peak per-sample change of
// 2*delta/N after IFFT. For int16 rounding to preserve it, delta must be >= N/2.
// Round-trip error per bin ≈ sqrt(N)*0.5 ≈ 22.6 for N=2048.
// QIM_STEP = 2048 gives peak per-sample change of 2, and decision margin
// of 1024 - 23 ≈ 1001 (very safe). This is what "higher bits" means:
// the perturbation lives well above the machine-error floor of the FT.

static constexpr int FRAME_SIZE = 2048;
static constexpr double QIM_STEP = 2048.0;

// Minimum per-frame RMS in dB for a frame to be a valid carrier.
// Quiet frames have bin magnitudes too small for QIM to survive round-trip.
// Frame RMS is computed in time-domain so it's invariant to QIM perturbation.
static constexpr float FRAME_THRESHOLD_DB = -30.0f;

// Header occupies first 96 samples (time-domain LSB, same as time-domain mode)
static constexpr int HEADER_NONCE_BITS = 64;
static constexpr int HEADER_LEN_BITS = 32;
static constexpr int HEADER_TOTAL = HEADER_NONCE_BITS + HEADER_LEN_BITS;

// --- Header embed/extract (time-domain, identical to steno.cpp) ---

template <typename T>
static void embed_header(std::vector<T>& samples, uint64_t nonce, uint32_t msgLen) {
    for (int i = 0; i < HEADER_NONCE_BITS; i++) {
        int bit = (nonce >> i) & 1;
        samples[i] = (samples[i] & ~static_cast<T>(1)) | static_cast<T>(bit);
    }
    for (int i = 0; i < HEADER_LEN_BITS; i++) {
        int bit = (msgLen >> i) & 1;
        samples[HEADER_NONCE_BITS + i] =
            (samples[HEADER_NONCE_BITS + i] & ~static_cast<T>(1)) | static_cast<T>(bit);
    }
}

template <typename T>
static uint32_t extract_msg_len(const std::vector<T>& samples) {
    uint32_t msgLen = 0;
    for (int i = 0; i < HEADER_LEN_BITS; i++) {
        msgLen |= (static_cast<uint32_t>(samples[HEADER_NONCE_BITS + i] & 1)) << i;
    }
    return msgLen;
}

// --- QIM encode/decode a single bit into a magnitude ---

// Embed: snap magnitude to nearest QIM grid point encoding the desired bit.
//   grid has spacing 2*Q. Even indices encode 0, odd indices encode 1.
static double qim_embed(double mag, int bit) {
    double Q = QIM_STEP;
    double half_grid = 2.0 * Q;
    // Quantize to nearest multiple of half_grid
    double base = std::round(mag / half_grid) * half_grid;
    // Candidate for bit 0: base, candidate for bit 1: base + Q
    double c0 = base;
    double c1 = base + Q;
    // Pick the candidate that matches the desired bit AND is closest to mag
    if (bit == 0) {
        // Closest even-index grid point
        if (std::abs(mag - c0) <= std::abs(mag - (c0 - half_grid)))
            return c0;
        else
            return c0 - half_grid;
    } else {
        // Closest odd-index grid point
        if (std::abs(mag - c1) <= std::abs(mag - (c1 - half_grid)))
            return c1;
        else
            return c1 - half_grid;
    }
}

// Decode: determine which QIM grid point is nearest.
static int qim_decode(double mag) {
    double Q = QIM_STEP;
    // Index in the Q-spaced grid
    long idx = std::lround(mag / Q);
    return static_cast<int>(((idx % 2) + 2) % 2); // 0 or 1
}

// --- Bin pool: which frequency bins in a frame are valid carriers ---

struct BinPos {
    int frame;  // frame index (0-based)
    int bin;    // frequency bin within the frame (1 to N/2-1)
};

// Compute per-frame RMS in time domain and return which frames are loud enough.
// This is invariant to QIM perturbation (which changes < 0.01% of frame energy),
// so encode and decode produce the same frame set.
static std::vector<bool> classify_frames(const std::vector<int16_t>& samples,
                                          int data_start, int num_frames) {
    float reference = 32768.0f;
    std::vector<bool> loud(num_frames, false);

    for (int f = 0; f < num_frames; f++) {
        int offset = data_start + f * FRAME_SIZE;
        double sum = 0.0;
        for (int i = 0; i < FRAME_SIZE; i++) {
            double s = static_cast<double>(samples[offset + i]);
            sum += s * s;
        }
        float rms = std::sqrt(static_cast<float>(sum / FRAME_SIZE));
        float db = (rms < 1e-10f) ? -100.0f : 20.0f * std::log10(rms / reference);
        loud[f] = (db > FRAME_THRESHOLD_DB);
    }
    return loud;
}

// Build the list of (frame, bin) positions from loud frames only.
// Excludes DC (bin 0) and Nyquist (bin N/2) since they have no conjugate partner.
static std::vector<BinPos> build_bin_pool(const std::vector<bool>& loud_frames,
                                           int num_frames) {
    std::vector<BinPos> pool;
    int half = FRAME_SIZE / 2;

    for (int f = 0; f < num_frames; f++) {
        if (!loud_frames[f]) continue;
        for (int b = 1; b < half; b++) {
            pool.push_back({f, b});
        }
    }
    return pool;
}

// --- Encode ---

EncodeResult steno_freq_encode(Wav& wav, const std::string& message) {
    EncodeResult result{false, ""};
    uint32_t msgLen = static_cast<uint32_t>(message.size());
    uint32_t totalBitsNeeded = msgLen * 8;

    std::visit([&](auto& samples) {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        bool is_8bit = std::is_same_v<T, uint8_t>;

        // Frequency-domain mode only supports 16-bit PCM
        if (is_8bit) return;

        int N = static_cast<int>(samples.size());
        if (N < HEADER_TOTAL + FRAME_SIZE) return;

        // 1. Generate nonce + embed header in time domain
        std::random_device rd;
        uint64_t nonce = (static_cast<uint64_t>(rd()) << 32) | rd();
        embed_header(samples, nonce, msgLen);

        // 2. Carve samples after header into fixed-size frames
        int data_start = HEADER_TOTAL;
        int data_len = N - data_start;
        int num_frames = data_len / FRAME_SIZE; // partial tail frame is dropped
        if (num_frames == 0) return;

        // 3. Classify frames by time-domain RMS (skip quiet frames)
        auto& int16_samples = std::get<std::vector<int16_t>>(wav.data.RawChannelSamples);
        auto loud = classify_frames(int16_samples, data_start, num_frames);

        // 4. FFT only loud frames (but allocate for all to keep indexing simple)
        std::vector<std::vector<FFT::Complex>> spectra(num_frames);
        for (int f = 0; f < num_frames; f++) {
            if (!loud[f]) continue;
            int offset = data_start + f * FRAME_SIZE;
            spectra[f].resize(FRAME_SIZE);
            for (int i = 0; i < FRAME_SIZE; i++) {
                spectra[f][i] = FFT::Complex(static_cast<double>(samples[offset + i]), 0.0);
            }
            FFT::forward(spectra[f]);
        }

        // 5. Build and shuffle bin pool (loud frames only)
        auto pool = build_bin_pool(loud, num_frames);
        if (pool.size() < totalBitsNeeded) return;

        uint64_t seed = derive_seed(nonce, wav.fmt);
        Xoshiro256ss prng;
        prng.seed_from(seed);
        for (int i = static_cast<int>(pool.size()) - 1; i > 0; i--) {
            int j = static_cast<int>(prng.next_bounded(i + 1));
            std::swap(pool[i], pool[j]);
        }

        // 6. Flatten message to bits
        std::vector<int> msgBits;
        msgBits.reserve(totalBitsNeeded);
        for (uint32_t byteIdx = 0; byteIdx < msgLen; byteIdx++) {
            uint8_t ch = static_cast<uint8_t>(message[byteIdx]);
            for (int b = 0; b < 8; b++) {
                msgBits.push_back((ch >> b) & 1);
            }
        }

        // 7. QIM-embed bits into selected frequency bins
        for (uint32_t i = 0; i < totalBitsNeeded; i++) {
            const BinPos& pos = pool[i];
            FFT::Complex& coeff = spectra[pos.frame][pos.bin];
            double mag = std::abs(coeff);
            double phase = std::arg(coeff);

            double new_mag = qim_embed(mag, msgBits[i]);
            // Preserve sign: keep magnitude non-negative
            if (new_mag < 0) new_mag = 0;
            coeff = std::polar(new_mag, phase);

            // Mirror conjugate bin to keep IFFT output real
            int conj_bin = FRAME_SIZE - pos.bin;
            spectra[pos.frame][conj_bin] = std::conj(coeff);
        }

        // 8. IFFT each modified frame and write back, clamping to int16 range
        for (int f = 0; f < num_frames; f++) {
            if (!loud[f]) continue;
            FFT::inverse(spectra[f]);
            int offset = data_start + f * FRAME_SIZE;
            for (int i = 0; i < FRAME_SIZE; i++) {
                double val = std::round(spectra[f][i].real());
                if (val > 32767.0) val = 32767.0;
                if (val < -32768.0) val = -32768.0;
                samples[offset + i] = static_cast<int16_t>(val);
            }
        }

        // Re-embed header (IFFT of frame 0 may have overwritten header samples
        // if data_start < FRAME_SIZE, but header is before data_start so it's safe)
        embed_header(samples, nonce, msgLen);

        result.success = true;
        result.cipher = hex_encode(nonce);
    }, wav.data.RawChannelSamples);

    return result;
}

// --- Decode ---

std::string steno_freq_decode(const Wav& wav, const std::string& cipher) {
    if (cipher.size() != 16) return "";
    uint64_t nonce = hex_decode(cipher);

    return std::visit([&](const auto& samples) -> std::string {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        bool is_8bit = std::is_same_v<T, uint8_t>;

        if (is_8bit) return "";

        int N = static_cast<int>(samples.size());
        if (N < HEADER_TOTAL + FRAME_SIZE) return "";

        // 1. Extract message length from time-domain header
        uint32_t msgLen = extract_msg_len(samples);
        if (msgLen == 0 || msgLen > 10 * 1024 * 1024) return "";
        uint32_t totalBitsNeeded = msgLen * 8;

        // 2. Classify frames and FFT loud ones
        int data_start = HEADER_TOTAL;
        int data_len = N - data_start;
        int num_frames = data_len / FRAME_SIZE;
        if (num_frames == 0) return "";

        auto& int16_samples = std::get<std::vector<int16_t>>(wav.data.RawChannelSamples);
        auto loud = classify_frames(int16_samples, data_start, num_frames);

        std::vector<std::vector<FFT::Complex>> spectra(num_frames);
        for (int f = 0; f < num_frames; f++) {
            if (!loud[f]) continue;
            int offset = data_start + f * FRAME_SIZE;
            spectra[f].resize(FRAME_SIZE);
            for (int i = 0; i < FRAME_SIZE; i++) {
                spectra[f][i] = FFT::Complex(static_cast<double>(samples[offset + i]), 0.0);
            }
            FFT::forward(spectra[f]);
        }

        // 3. Rebuild and shuffle bin pool (loud frames only, identical to encode)
        auto pool = build_bin_pool(loud, num_frames);
        if (pool.size() < totalBitsNeeded) return "";

        uint64_t seed = derive_seed(nonce, wav.fmt);
        Xoshiro256ss prng;
        prng.seed_from(seed);
        for (int i = static_cast<int>(pool.size()) - 1; i > 0; i--) {
            int j = static_cast<int>(prng.next_bounded(i + 1));
            std::swap(pool[i], pool[j]);
        }

        // 4. QIM-decode bits from selected bins
        std::vector<int> msgBits;
        msgBits.reserve(totalBitsNeeded);
        for (uint32_t i = 0; i < totalBitsNeeded; i++) {
            const BinPos& pos = pool[i];
            double mag = std::abs(spectra[pos.frame][pos.bin]);
            msgBits.push_back(qim_decode(mag));
        }

        // 5. Reassemble bytes
        std::string result;
        result.reserve(msgLen);
        for (uint32_t i = 0; i < msgLen; i++) {
            uint8_t ch = 0;
            for (int b = 0; b < 8; b++) {
                ch |= static_cast<uint8_t>(msgBits[i * 8 + b]) << b;
            }
            result.push_back(static_cast<char>(ch));
        }
        return result;
    }, wav.data.RawChannelSamples);
}
