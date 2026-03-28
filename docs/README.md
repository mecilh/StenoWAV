# StenoWav Documentation

Welcome to StenoWav — an audio steganography framework that hides data inside WAV files using Quantisation Index Modulation (QIM) with a packet-based cipher system.

## Quick Start

1. **New to StenoWav?** Start with [stenowav_guide.pdf](stenowav_guide.pdf)
   - Plain-English explanation of how it works
   - Step-by-step configuration guide
   - Practical code examples
   - Common questions answered

2. **Want the full technical story?** See [stenowav_technical.pdf](stenowav_technical.pdf)
   - Complete mathematical formalisation with proofs
   - QIM theory and cipher derivation
   - Security analysis and capacity calculations
   - Detailed algorithm specifications
   - Full API reference

## What Is StenoWav?

StenoWav lets you embed secret messages into audio signals so that:
- The embedded data survives 16-bit WAV compression
- Only holders of the cryptographic key can extract it
- The modifications are imperceptible to human hearing
- The data placement is intelligence-driven (RMS-adaptive) for robustness

It works in both the time domain (directly modifying samples) and frequency domain (embedding in magnitude spectra), with optional dither for additional security.

## Key Capabilities

- **Dual-domain embedding**: time-domain direct QIM or frequency-domain magnitude modification
- **Intelligent bit placement**: RMS-adaptive selection ensures payload lives in signals loud enough to mask the changes
- **Self-contained extraction**: encoded messages carry their own metadata, decoder only needs the watermarked audio and the key
- **Perceptual safety**: all modifications stay inaudible (SNR > 30 dB)
- **Optional dither**: raise the noise floor to prevent statistical attacks
- **Configurable**: every parameter is tunable via `StenoConfig`

## Framework Structure

```
Steno (public API)
├── time-domain path
│   ├── RMS-adaptive placement
│   └── QIM + cipher embedding
└── frequency-domain path
    ├── FFT analysis
    ├── Magnitude embedding (non-negative QIM)
    └── IFFT synthesis
```

Each layer — from WAV I/O to packet generation to dither — is independent and well-tested. The full test suite: **118 tests passing**, including integration tests on real audio files.

## For Developers

See the Rust documentation in the main crate:

```bash
cd /repo
cargo doc --open
```

- `lib.rs` — module overview
- `steno.rs` — public API (`Steno::encode`, `Steno::decode`)
- `embed.rs` — dual-domain embedding engine
- `packet.rs` — packet architecture and cipher system
- `qim.rs` — quantisation index modulation primitives
- `wav.rs` — WAV I/O (pure Rust, no external deps)
- `fft.rs` — FFI bindings to C math library

All tests: `cargo test --lib`

## Questions?

- **"How do I embed a message?"** → See the User Guide, Section 3
- **"Why does my decoder fail?"** → Check the Key Management advice in the User Guide
- **"What parameters should I use?"** → See the Configuration Guide and Practical Guidelines in the User Guide
- **"Can it survive MP3 compression?"** → See Future Directions in both documents

---

*StenoWav: hide secrets in music. Enterprise-level framework, built with love.*
