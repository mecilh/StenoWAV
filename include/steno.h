#ifndef STENO_H
#define STENO_H

#include "wav.h"
#include <string>

// Encode a text message into the LSBs of WAV audio samples.
// Returns false if the message is too large for the audio file.
bool steno_encode(Wav& wav, const std::string& message);

// Decode a hidden message from the LSBs of WAV audio samples.
std::string steno_decode(const Wav& wav);

#endif
