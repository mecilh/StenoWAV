#ifndef STENO_FREQ_H
#define STENO_FREQ_H

#include "wav.h"
#include <string>

struct EncodeResult; // forward-declared from steno.h

// Frequency-domain scatter-embed steganography.
// Embeds data into FFT bin magnitudes via QIM (Quantization Index Modulation).
// Uses higher bits (large quantization step) because FFT round-trip precision
// loss destroys lower-significance information.

EncodeResult steno_freq_encode(Wav& wav, const std::string& message);
std::string  steno_freq_decode(const Wav& wav, const std::string& cipher);

#endif
