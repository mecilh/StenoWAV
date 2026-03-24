#ifndef STENO_H
#define STENO_H

#include "wav.h"
#include <string>

struct EncodeResult {
    bool success;
    std::string cipher;  // hex-encoded 64-bit nonce
};

// Encode a message into scattered WAV samples using PRNG-linked traversal.
// Returns the auto-generated cipher needed for decoding.
EncodeResult steno_encode(Wav& wav, const std::string& message);

// Decode a hidden message using the cipher from encoding.
std::string steno_decode(const Wav& wav, const std::string& cipher);

#endif
