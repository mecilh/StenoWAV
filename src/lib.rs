//! # StenoWav Framework
//!
//! Audio steganography framework based on Quantisation Index Modulation (QIM).
//!
//! Embeds arbitrary data into audio signals across both time and frequency
//! domains using a packet-based architecture with mask/cipher obfuscation.
//!
//! ## Architecture
//!
//! - `steno` — **Public API**: `Steno::encode` / `Steno::decode` with `StenoConfig`
//! - `wav` — Read/write 16-bit PCM WAV files
//! - `fft` — FFT/IFFT via C math library (radix-2 Cooley-Tukey)
//! - `qim` — QIM embed/extract engine with configurable step size
//! - `packet` — Packet-based encoding with mask/cipher obfuscation
//! - `cipher` — Cipher key derivation from mask samples
//! - `embed` — Dual-domain embedding engine with RMS-adaptive placement
//! - `dither` — Optional noise dither and perceptual quality metrics

pub mod cipher;
pub mod dither;
pub mod embed;
pub mod fft;
pub mod packet;
pub mod qim;
pub mod steno;
pub mod wav;
