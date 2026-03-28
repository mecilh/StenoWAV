# StenoWav

An audio steganography framework that hides data inside WAV files. Because apparently regular encryption wasn't paranoid enough.

## What It Does

StenoWav embeds arbitrary data into audio signals using [Quantisation Index Modulation](https://en.wikipedia.org/wiki/Quantization_index_modulation) (QIM) — a fancy way of saying "nudge samples onto a grid and hope nobody notices." It works in both the time domain and the frequency domain, because why settle for one way to be undetectable when you can have two.

The embedded data survives the 16-bit WAV round-trip, extraction is impossible without the cryptographic key, and the modifications are inaudible to human ears. We checked. 118 tests checked. The math checked. Your ears won't check because that's the whole point.

## Quick Start

```rust
use stenowav::steno::{Steno, StenoConfig};

let key = [42u8; 32]; // please use a real key in production, we beg you
let config = StenoConfig::default();

// Hide your secrets
Steno::encode("song.wav", "totally_normal_song.wav", b"launch codes", &key, &config).unwrap();

// Retrieve your secrets
let recovered = Steno::decode("totally_normal_song.wav", &key, &config).unwrap();
assert_eq!(recovered, b"launch codes");
```

Two function calls. That's it. Everything else — the packet system, the cipher dance, the RMS analysis, the non-negative QIM constraints for frequency magnitudes — is someone else's problem now. Yours, specifically, if you read the source.

## How It Works (The Short Version)

1. **QIM** quantises audio samples onto a lattice. Even lattice points encode `0`, odd ones encode `1`. The decoder rounds and checks parity. It's almost offensively simple.
2. **Packets** group 9 pseudo-random sample indices: 8 mask samples produce a cipher key, 1 data region carries payload XORed with that cipher. Without the seed, you don't know which samples to look at. Good luck.
3. **RMS-adaptive placement** analyses signal power and only embeds in segments loud enough to mask the modifications. It won't whisper your secrets into a quiet passage like an amateur.
4. **Frequency-domain** path does FFT, embeds in magnitude bins using non-negative QIM (because magnitudes can't be negative, a lesson we learned the hard way), then IFFTs back.
5. **Dither** optionally perturbs every non-payload sample with tiny noise, so the whole signal looks uniformly "slightly noisy" instead of having suspicious pockets of grid-aligned values next to pristine ones.
6. A **self-contained header** (first 7 packets) carries metadata, so the decoder needs nothing but the watermarked audio and the key. No sidecar files. No "oh wait, what was the packet count again."

## Features Nobody Asked For But You'll Appreciate Anyway

- **Dual-domain embedding** — time or frequency, your call
- **Self-contained decoder** — embedded STEN header means no metadata files to lose
- **Pure Rust + C** — no Python. Not even a little. The C is just for the FFT because Cooley and Tukey didn't write Rust
- **Every parameter is tuneable** — delta, RMS thresholds, FFT window size, bin ranges, dither intensity.
- **Perceptual safety** — SNR > 30 dB with embedding, > 50 dB with pure dither. CD quality considers > 20 dB "transparent." We're not subtle about being subtle

## Configuration

```rust
// Defaults work fine. But if you must tinker:
let config = StenoConfig::frequency()
    .with_freq_delta(0.08)
    .with_dither();
```

`StenoConfig` is a flat struct with builder methods. No nested config-of-configs-of-configs. Every field has a sane default. `validate()` tells you if you've done something foolish before the encoder does.

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `domain` | Time | Time-domain or frequency-domain embedding |
| `time_delta` | 0.005 | QIM step size for time domain |
| `freq_delta` | 0.05 | QIM step size for frequency domain |
| `rms_high` | 0.10 | RMS threshold for "loud enough" segments |
| `rms_low` | 0.01 | RMS threshold for "too quiet, skip" segments |
| `dither_enabled` | false | Whether to add noise to non-payload samples |
| `dither_intensity` | ~6.1e-5 | 2 LSBs in normalised space. Inaudible. Probably. |
| `channel` | 0 | Which audio channel to embed in |

## Architecture

```
src/
├── steno.rs    — Public API. The part you actually use.
├── embed.rs    — Dual-domain engine, RMS analysis, header system
├── packet.rs   — Packet construction, index selection, stream framing
├── cipher.rs   — CipherKey derivation, mod-4 QIM, the XOR that keeps secrets secret
├── dither.rs   — Noise generation, quality metrics (SNR, spectral diff)
├── qim.rs      — The core algorithm. 30 lines that do all the real work.
├── fft.rs      — Rust FFI bindings to the C math library
├── wav.rs      — WAV reader/writer. Pure Rust. No dependencies. It simply works.
└── lib.rs      — Module declarations. Riveting.

csrc/
├── math_ops.c  — Radix-2 Cooley-Tukey FFT, IFFT, RMS, complex helpers
└── math_ops.h  — Header file. It declares things.
```

Each layer trusts the one below it. `steno` calls `embed`, `embed` calls `packet`, `packet` calls `cipher` and `qim`, `qim` does maths. It's turtles all the way down, except the bottom turtle is C.

## Building

```bash
cd repo
cargo build --release
cargo test  # all 118, or don't bother shipping
```

Requires a C compiler (for the FFT library).

## Documentation

- **[User Guide](docs/stenowav_guide.pdf)** — 12 pages. Plain English. Code examples. A glossary, because we're civilised.
- **[Technical Paper](docs/stenowav_technical.pdf)** — 13 pages. Theorems. Proofs. The kind of document you cite in a bibliography to look smart.

## Known Limitations

- **WAV only** — no MP3, no FLAC, no Ogg. The i16 round-trip is hard enough without lossy compression getting involved. Compression robustness is future work, assuming the future cooperates.
- **Power-of-2 FFT** — the C library requires it. Tail samples that don't fill a window are skipped. Your 3-second voice memo is fine. Probably.
- **Capacity** — each packet uses ~24 samples for 2 bytes of payload. You need roughly 100x more samples than you're embedding. A 3-minute song at 44.1kHz gives you millions of samples, so this is only a problem if you're trying to hide a novel inside a ringtone.
- **No streaming mode** — whole-file processing only. Real-time embedding is on the roadmap, filed under "ambitious."

## Future Work

- AWGN simulation sweeps to find optimal parameters (all the knobs are exposed, we just haven't turned them all yet)
- Compression robustness (MP3, AAC)
- Variable packet sizes beyond 9
- Streaming / real-time mode
- A Python analysis script for parameter optimisation (the one exception to the "no Python" rule, and it will live in exile outside `/repo`)

## Dependencies

- `rand` 0.9 + `rand_chacha` 0.9 — for ChaCha20 CSPRNG. Because `rand::thread_rng()` is not reproducible and reproducibility is the entire point of a decoder.
- `cc` (build only) — compiles the C math library. The build system equivalent of "it's not much, but it's honest work."

No runtime dependencies beyond `rand`. No `serde`. No `tokio`. No framework-of-the-week. Just maths and stubbornness.

## License

Not yet decided. Don't steal it in the meantime. Or do. We hid it in a WAV file anyway, good luck finding it.

---
