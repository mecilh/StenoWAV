#include "steno.h"
#include <cstdint>

bool steno_encode(Wav& wav, const std::string& message) {
    uint32_t msgLen = static_cast<uint32_t>(message.size());
    uint32_t bitsNeeded = 32 + msgLen * 8;

    return std::visit([&](auto& samples) -> bool {
        if (bitsNeeded > static_cast<uint32_t>(samples.size())) {
            return false;
        }

        // Embed 32-bit message length in first 32 sample LSBs
        for (uint32_t i = 0; i < 32; i++) {
            int bit = (msgLen >> i) & 1;
            samples[i] = (samples[i] & ~1) | bit;
        }

        // Embed message bytes, 1 bit per sample, LSB-first per byte
        uint32_t sampleIdx = 32;
        for (uint32_t byteIdx = 0; byteIdx < msgLen; byteIdx++) {
            uint8_t ch = static_cast<uint8_t>(message[byteIdx]);
            for (int bitIdx = 0; bitIdx < 8; bitIdx++) {
                int bit = (ch >> bitIdx) & 1;
                samples[sampleIdx] = (samples[sampleIdx] & ~1) | bit;
                sampleIdx++;
            }
        }

        return true;
    }, wav.data.RawChannelSamples);
}

std::string steno_decode(const Wav& wav) {
    return std::visit([&](const auto& samples) -> std::string {
        if (samples.size() < 32) {
            return "";
        }

        // Extract 32-bit message length from first 32 sample LSBs
        uint32_t msgLen = 0;
        for (uint32_t i = 0; i < 32; i++) {
            msgLen |= (static_cast<uint32_t>(samples[i] & 1)) << i;
        }

        uint32_t bitsNeeded = 32 + msgLen * 8;
        if (bitsNeeded > static_cast<uint32_t>(samples.size())) {
            return "";
        }

        // Extract message bytes
        std::string result;
        result.reserve(msgLen);
        uint32_t sampleIdx = 32;
        for (uint32_t byteIdx = 0; byteIdx < msgLen; byteIdx++) {
            uint8_t ch = 0;
            for (int bitIdx = 0; bitIdx < 8; bitIdx++) {
                ch |= (static_cast<uint8_t>(samples[sampleIdx] & 1)) << bitIdx;
                sampleIdx++;
            }
            result.push_back(static_cast<char>(ch));
        }

        return result;
    }, wav.data.RawChannelSamples);
}
