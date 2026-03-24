#include "steno.h"
#include "math_steno.h"
#include <cstdint>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>

// --- SplitMix64: used for seed initialization ---

static uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// --- Xoshiro256** PRNG ---

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
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Uniform random in [0, bound)
    uint32_t next_bounded(uint32_t bound) {
        return static_cast<uint32_t>(next() % bound);
    }
};

// --- Seed derivation ---

static uint64_t derive_seed(uint64_t nonce, const Wav::FmtChunk& fmt) {
    uint64_t mixed = nonce;
    mixed ^= static_cast<uint64_t>(fmt.AudioFormat) << 0;
    mixed ^= static_cast<uint64_t>(fmt.NbrChannel) << 16;
    mixed ^= static_cast<uint64_t>(fmt.Frequency) << 32;
    mixed ^= static_cast<uint64_t>(fmt.BitsPerSample) << 48;
    // Run through splitmix to avalanche
    return splitmix64(mixed);
}

// --- Hex encoding/decoding ---

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

// --- Bit budget based on local loudness ---

template <typename T>
static int bit_budget(const std::vector<T>& samples, int idx, bool is_8bit) {
    if (is_8bit) return 1;  // 8-bit audio: always 1 bit only

    float reference = 32768.0f;
    float rms = MathSteno::TimeDomain::rms_window(samples, idx, 1024);
    float db = MathSteno::TimeDomain::rms_to_db(rms, reference);

    if (db > -10.0f) return 3;
    if (db > -20.0f) return 2;
    return 1;
}

// Overload for uint8_t (always 1 bit, reference 128)
static int bit_budget(const std::vector<uint8_t>& /*samples*/, int /*idx*/, bool) {
    return 1;
}

// --- Select which bit positions to use within the budget ---
// Only uses bits 0..budget-1 so the max perturbation is bounded by the budget.

static std::vector<int> select_bit_positions(int budget) {
    std::vector<int> positions;
    positions.reserve(budget);
    for (int i = 0; i < budget; i++) {
        positions.push_back(i);
    }
    return positions;
}

// --- Header embedding/extraction (samples 0-95, LSB only) ---

static constexpr int HEADER_NONCE_BITS = 64;
static constexpr int HEADER_LEN_BITS = 32;
static constexpr int HEADER_TOTAL = HEADER_NONCE_BITS + HEADER_LEN_BITS; // 96

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

// --- Core encode/decode ---

EncodeResult steno_encode(Wav& wav, const std::string& message, bool dither) {
    EncodeResult result{false, ""};
    uint32_t msgLen = static_cast<uint32_t>(message.size());
    uint32_t totalBitsNeeded = msgLen * 8;

    std::visit([&](auto& samples) {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        bool is_8bit = std::is_same_v<T, uint8_t>;

        if (static_cast<int>(samples.size()) < HEADER_TOTAL + 1) {
            return; // too small
        }

        // 1. Generate random nonce
        std::random_device rd;
        uint64_t nonce = (static_cast<uint64_t>(rd()) << 32) | rd();

        // 2. Embed header (nonce + message length)
        embed_header(samples, nonce, msgLen);

        // 3. Build valid-position pool (indices >= HEADER_TOTAL only)
        auto all_valid = MathSteno::TimeDomain::validate_position_time_domain(
            samples, 1024, -30.0f);

        std::vector<int> pool;
        pool.reserve(all_valid.size());
        for (int idx : all_valid) {
            if (idx >= HEADER_TOTAL) {
                pool.push_back(idx);
            }
        }

        if (pool.empty()) {
            return;
        }

        // 4. Seed PRNG and shuffle pool (Fisher-Yates)
        uint64_t seed = derive_seed(nonce, wav.fmt);
        Xoshiro256ss prng;
        prng.seed_from(seed);

        for (int i = static_cast<int>(pool.size()) - 1; i > 0; i--) {
            int j = static_cast<int>(prng.next_bounded(i + 1));
            std::swap(pool[i], pool[j]);
        }

        // 5. Flatten message to bits (LSB-first per byte)
        std::vector<int> msgBits;
        msgBits.reserve(totalBitsNeeded);
        for (uint32_t byteIdx = 0; byteIdx < msgLen; byteIdx++) {
            uint8_t ch = static_cast<uint8_t>(message[byteIdx]);
            for (int b = 0; b < 8; b++) {
                msgBits.push_back((ch >> b) & 1);
            }
        }

        // 6. Embed data into scatter positions with adaptive bit depth
        uint32_t bitCursor = 0;

        for (int idx : pool) {
            if (bitCursor >= totalBitsNeeded) break;

            int budget = bit_budget(samples, idx, is_8bit);
            auto bitPositions = select_bit_positions(budget);

            for (int bp : bitPositions) {
                if (bitCursor >= totalBitsNeeded) break;
                T mask = static_cast<T>(1) << bp;
                T dataBit = static_cast<T>(msgBits[bitCursor]) << bp;
                samples[idx] = (samples[idx] & ~mask) | dataBit;
                bitCursor++;
            }
        }

        if (bitCursor < totalBitsNeeded) {
            return; // not enough capacity
        }

        // 7. Optional dither: add LSB noise to ALL samples to mask embed positions
        if (dither) {
            Xoshiro256ss ditherPrng;
            ditherPrng.seed_from(nonce ^ 0xA5A5A5A5A5A5A5A5ULL);
            for (size_t i = 0; i < samples.size(); i++) {
                // Flip bit 0 with 50% probability on non-embedded samples
                // For embedded samples this just adds another layer of noise
                if (ditherPrng.next() & 1) {
                    samples[i] ^= static_cast<T>(1);
                }
            }
            // Re-embed the header so it survives the dither
            embed_header(samples, nonce, msgLen);
            // Re-embed the data bits on top of dither
            bitCursor = 0;
            // Re-seed and re-shuffle a fresh pool copy for deterministic replay
            Xoshiro256ss prng2;
            prng2.seed_from(seed);
            std::vector<int> pool2 = pool;
            // pool was already shuffled in-place, reuse it directly
            for (int idx : pool) {
                if (bitCursor >= totalBitsNeeded) break;
                int b = bit_budget(samples, idx, is_8bit);
                auto bps = select_bit_positions(b);
                for (int bp : bps) {
                    if (bitCursor >= totalBitsNeeded) break;
                    T mask = static_cast<T>(1) << bp;
                    T dataBit = static_cast<T>(msgBits[bitCursor]) << bp;
                    samples[idx] = (samples[idx] & ~mask) | dataBit;
                    bitCursor++;
                }
            }
        }

        result.success = true;
        result.cipher = hex_encode(nonce);
    }, wav.data.RawChannelSamples);

    return result;
}

std::string steno_decode(const Wav& wav, const std::string& cipher) {
    if (cipher.size() != 16) return "";

    uint64_t nonce = hex_decode(cipher);

    return std::visit([&](const auto& samples) -> std::string {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        bool is_8bit = std::is_same_v<T, uint8_t>;

        if (static_cast<int>(samples.size()) < HEADER_TOTAL + 1) {
            return "";
        }

        // 1. Extract message length from header
        uint32_t msgLen = extract_msg_len(samples);

        // Sanity check
        if (msgLen == 0 || msgLen > 10 * 1024 * 1024) {
            return "";
        }

        uint32_t totalBitsNeeded = msgLen * 8;

        // 2. Rebuild valid-position pool
        auto all_valid = MathSteno::TimeDomain::validate_position_time_domain(
            samples, 1024, -30.0f);

        std::vector<int> pool;
        pool.reserve(all_valid.size());
        for (int idx : all_valid) {
            if (idx >= HEADER_TOTAL) {
                pool.push_back(idx);
            }
        }

        if (pool.empty()) {
            return "";
        }

        // 3. Seed PRNG and reproduce the same shuffle
        uint64_t seed = derive_seed(nonce, wav.fmt);
        Xoshiro256ss prng;
        prng.seed_from(seed);

        for (int i = static_cast<int>(pool.size()) - 1; i > 0; i--) {
            int j = static_cast<int>(prng.next_bounded(i + 1));
            std::swap(pool[i], pool[j]);
        }

        // 4. Extract data bits from scatter positions
        std::vector<int> msgBits;
        msgBits.reserve(totalBitsNeeded);

        for (int idx : pool) {
            if (static_cast<uint32_t>(msgBits.size()) >= totalBitsNeeded) break;

            int budget = bit_budget(samples, idx, is_8bit);
            auto bitPositions = select_bit_positions(budget);

            for (int bp : bitPositions) {
                if (static_cast<uint32_t>(msgBits.size()) >= totalBitsNeeded) break;
                int bit = (samples[idx] >> bp) & 1;
                msgBits.push_back(bit);
            }
        }

        if (static_cast<uint32_t>(msgBits.size()) < totalBitsNeeded) {
            return "";
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
