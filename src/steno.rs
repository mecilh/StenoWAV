//! Public API for the StenoWav framework.
//!
//! This module provides the top-level `Steno` struct — the primary interface
//! for encoding and decoding data in WAV audio files. It wraps the internal
//! embedding engine, WAV I/O, and configuration into a clean, minimal API.
//!
//! ## Quick start
//!
//! ```no_run
//! use stenowav::steno::{Steno, StenoConfig};
//!
//! // Encode a secret message into a WAV file
//! let key = [42u8; 32];
//! let config = StenoConfig::default();
//! Steno::encode("input.wav", "output.wav", b"secret message", &key, &config).unwrap();
//!
//! // Decode it back
//! let recovered = Steno::decode("output.wav", &key, &config).unwrap();
//! assert_eq!(recovered, b"secret message");
//! ```
//!
//! ## Configuration
//!
//! [`StenoConfig`] exposes all tuneable parameters with sane defaults:
//!
//! - **Domain**: time or frequency (default: time)
//! - **QIM deltas**: step sizes for time and frequency embedding
//! - **RMS thresholds**: power tier boundaries for adaptive placement
//! - **Frequency config**: FFT window, bin range
//! - **Dither**: optional noise to mask QIM patterns
//! - **Packet size**: 9 samples per packet (designed for future expansion)
//!
//! All defaults work out of the box for typical 16-bit 44.1kHz audio.

use crate::dither::DitherConfig;
use crate::embed::{self, EmbedConfig, EmbedDomain, FreqConfig, RmsConfig, RmsThresholds};
use crate::wav::{self, WavData, WavError};
use std::fmt;
use std::path::Path;

// ══════════════════════════════════════════════════════════════════
// Error type
// ══════════════════════════════════════════════════════════════════

/// Errors from the Steno public API.
#[derive(Debug)]
pub enum StenoError {
    /// WAV file I/O error.
    Wav(WavError),
    /// Embedding/extraction error.
    Embed(embed::EmbedError),
    /// Invalid configuration.
    Config(String),
}

impl fmt::Display for StenoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StenoError::Wav(e) => write!(f, "WAV error: {}", e),
            StenoError::Embed(e) => write!(f, "embed error: {}", e),
            StenoError::Config(msg) => write!(f, "config error: {}", msg),
        }
    }
}

impl std::error::Error for StenoError {}

impl From<WavError> for StenoError {
    fn from(e: WavError) -> Self {
        StenoError::Wav(e)
    }
}

impl From<embed::EmbedError> for StenoError {
    fn from(e: embed::EmbedError) -> Self {
        StenoError::Embed(e)
    }
}

// ══════════════════════════════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════════════════════════════

/// User-facing configuration for the StenoWav framework.
///
/// All fields have sane defaults that work out of the box for typical
/// 16-bit 44.1kHz audio. Adjust individual fields for specific needs
/// or simulation sweeps.
///
/// ## Defaults
///
/// | Parameter | Default | Notes |
/// |-----------|---------|-------|
/// | domain | Time | Direct PCM embedding |
/// | time_delta | 0.005 | QIM step size for time domain |
/// | freq_delta | 0.05 | QIM step size for frequency domain |
/// | rms_high | 0.10 | Above this = High power tier |
/// | rms_low | 0.01 | Below this = Skip tier |
/// | rms_segment_size | 4096 | Samples per RMS segment |
/// | freq_window_size | 1024 | FFT window (power of 2) |
/// | freq_min_bin | 10 | Skip DC and very low freqs |
/// | freq_max_bin | 400 | Upper bin limit |
/// | dither_enabled | false | Optional noise masking |
/// | dither_intensity | ~2 LSBs | ~6.1e-5 in normalised space |
/// | channel | 0 | Which audio channel to embed in |
#[derive(Debug, Clone)]
pub struct StenoConfig {
    /// Which domain to embed data in.
    pub domain: EmbedDomain,

    /// QIM step size for time-domain embedding (PCM samples in \[-1, 1\]).
    /// Larger = more robust to noise, but more audible distortion.
    pub time_delta: f64,

    /// RMS threshold between Mid and High power tiers.
    /// Segments with RMS >= this are classified as High.
    pub rms_high: f64,

    /// RMS threshold between Skip and Mid power tiers.
    /// Segments with RMS < this are skipped entirely.
    pub rms_low: f64,

    /// Number of samples per segment for RMS computation.
    /// Smaller = finer-grained power classification.
    pub rms_segment_size: usize,

    /// FFT window size for frequency-domain embedding (must be power of 2).
    pub freq_window_size: usize,

    /// Lowest frequency bin index to embed in (skip DC and low freqs).
    pub freq_min_bin: usize,

    /// Highest frequency bin index to embed in (exclusive).
    pub freq_max_bin: usize,

    /// QIM step size for frequency-domain magnitudes.
    pub freq_delta: f64,

    /// Whether dither is enabled. Dither adds small random perturbations
    /// to non-payload samples, masking QIM lattice patterns.
    pub dither_enabled: bool,

    /// Maximum dither perturbation amplitude in normalised sample space.
    /// Default: ~2 LSBs for 16-bit PCM (~6.1e-5).
    pub dither_intensity: f64,

    /// Which audio channel to embed data in (0-indexed).
    /// For stereo files, 0 = left, 1 = right.
    pub channel: usize,
}

impl Default for StenoConfig {
    fn default() -> Self {
        let dither_default = DitherConfig::default();
        Self {
            domain: EmbedDomain::Time,
            time_delta: 0.005,
            rms_high: 0.10,
            rms_low: 0.01,
            rms_segment_size: 4096,
            freq_window_size: 1024,
            freq_min_bin: 10,
            freq_max_bin: 400,
            freq_delta: 0.05,
            dither_enabled: dither_default.enabled,
            dither_intensity: dither_default.intensity,
            channel: 0,
        }
    }
}

impl StenoConfig {
    /// Create a config for time-domain embedding with all defaults.
    pub fn time() -> Self {
        Self::default()
    }

    /// Create a config for frequency-domain embedding with all defaults.
    pub fn frequency() -> Self {
        Self {
            domain: EmbedDomain::Frequency,
            ..Self::default()
        }
    }

    /// Enable dither with the default intensity.
    pub fn with_dither(mut self) -> Self {
        self.dither_enabled = true;
        self
    }

    /// Set the time-domain QIM delta.
    pub fn with_time_delta(mut self, delta: f64) -> Self {
        self.time_delta = delta;
        self
    }

    /// Set the frequency-domain QIM delta.
    pub fn with_freq_delta(mut self, delta: f64) -> Self {
        self.freq_delta = delta;
        self
    }

    /// Set the audio channel to embed in.
    pub fn with_channel(mut self, channel: usize) -> Self {
        self.channel = channel;
        self
    }

    /// Convert to the internal `EmbedConfig` used by the embedding engine.
    fn to_embed_config(&self, seed: &[u8; 32]) -> EmbedConfig {
        EmbedConfig {
            domain: self.domain,
            time_delta: self.time_delta,
            rms: RmsConfig {
                segment_size: self.rms_segment_size,
                thresholds: RmsThresholds {
                    high: self.rms_high,
                    low: self.rms_low,
                },
            },
            freq: FreqConfig {
                window_size: self.freq_window_size,
                min_bin: self.freq_min_bin,
                max_bin: self.freq_max_bin,
                delta: self.freq_delta,
            },
            seed: *seed,
            dither: DitherConfig {
                enabled: self.dither_enabled,
                intensity: self.dither_intensity,
            },
            ..EmbedConfig::default()
        }
    }

    /// Validate the configuration, returning an error if anything is invalid.
    fn validate(&self) -> Result<(), StenoError> {
        if self.time_delta <= 0.0 || !self.time_delta.is_finite() {
            return Err(StenoError::Config(format!(
                "time_delta must be positive and finite, got {}",
                self.time_delta
            )));
        }
        if self.freq_delta <= 0.0 || !self.freq_delta.is_finite() {
            return Err(StenoError::Config(format!(
                "freq_delta must be positive and finite, got {}",
                self.freq_delta
            )));
        }
        if self.rms_low >= self.rms_high {
            return Err(StenoError::Config(format!(
                "rms_low ({}) must be less than rms_high ({})",
                self.rms_low, self.rms_high
            )));
        }
        if self.rms_segment_size == 0 {
            return Err(StenoError::Config(
                "rms_segment_size must be > 0".into(),
            ));
        }
        let ws = self.freq_window_size;
        if ws == 0 || (ws & (ws - 1)) != 0 {
            return Err(StenoError::Config(format!(
                "freq_window_size must be a power of 2, got {}",
                ws
            )));
        }
        if self.freq_min_bin >= self.freq_max_bin {
            return Err(StenoError::Config(format!(
                "freq_min_bin ({}) must be less than freq_max_bin ({})",
                self.freq_min_bin, self.freq_max_bin
            )));
        }
        if self.dither_intensity < 0.0 {
            return Err(StenoError::Config(format!(
                "dither_intensity must be >= 0, got {}",
                self.dither_intensity
            )));
        }
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════
// Steno — the public API
// ══════════════════════════════════════════════════════════════════

/// The primary interface to the StenoWav framework.
///
/// `Steno` provides two static methods — [`encode`](Steno::encode) and
/// [`decode`](Steno::decode) — that handle the full pipeline from file
/// to file (or file to data).
///
/// ## Example
///
/// ```no_run
/// use stenowav::steno::{Steno, StenoConfig};
///
/// let key = [0u8; 32]; // Your secret key (32 bytes)
/// let config = StenoConfig::default();
///
/// // Hide data in audio
/// Steno::encode("clean.wav", "watermarked.wav", b"Hello!", &key, &config).unwrap();
///
/// // Recover data from audio
/// let data = Steno::decode("watermarked.wav", &key, &config).unwrap();
/// assert_eq!(&data, b"Hello!");
/// ```
pub struct Steno;

impl Steno {
    /// Encode arbitrary data into a WAV audio file.
    ///
    /// Reads the input WAV, embeds `data` into the specified channel using
    /// QIM with the given key and config, and writes the watermarked audio
    /// to the output path. The output WAV preserves all original audio
    /// properties (sample rate, channels, etc.).
    ///
    /// # Arguments
    ///
    /// * `input_path` — Path to the source WAV file (16-bit PCM)
    /// * `output_path` — Path to write the watermarked WAV file
    /// * `data` — The byte payload to embed
    /// * `key` — 32-byte secret key (seed for CSPRNG index selection + cipher)
    /// * `config` — Encoding parameters (use `StenoConfig::default()` for sane defaults)
    ///
    /// # Errors
    ///
    /// Returns `StenoError` if:
    /// - The input WAV cannot be read or is not 16-bit PCM
    /// - The config is invalid
    /// - The selected channel doesn't exist
    /// - The signal is too short or too quiet to embed the data
    pub fn encode<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        data: &[u8],
        key: &[u8; 32],
        config: &StenoConfig,
    ) -> Result<(), StenoError> {
        config.validate()?;

        // Read input WAV
        let mut wav_data = wav::read_wav(&input_path)?;

        // Validate channel selection
        if config.channel >= wav_data.num_channels() {
            return Err(StenoError::Config(format!(
                "channel {} does not exist (file has {} channel{})",
                config.channel,
                wav_data.num_channels(),
                if wav_data.num_channels() == 1 { "" } else { "s" }
            )));
        }

        // Build internal config and embed
        let embed_config = config.to_embed_config(key);
        let samples = &mut wav_data.channels[config.channel];
        embed::encode_full(samples, data, &embed_config)?;

        // Write output WAV
        wav::write_wav(&output_path, &wav_data)?;

        Ok(())
    }

    /// Decode data from a watermarked WAV audio file.
    ///
    /// Reads the watermarked WAV, extracts the embedded header and data
    /// packets from the specified channel using the same key and config
    /// that were used for encoding.
    ///
    /// # Arguments
    ///
    /// * `watermarked_path` — Path to the watermarked WAV file
    /// * `key` — 32-byte secret key (must match the key used for encoding)
    /// * `config` — Decoding parameters (must match encoding config)
    ///
    /// # Errors
    ///
    /// Returns `StenoError` if:
    /// - The WAV file cannot be read
    /// - The config is invalid
    /// - The selected channel doesn't exist
    /// - The embedded header is invalid (wrong key, wrong config, or not watermarked)
    pub fn decode<P: AsRef<Path>>(
        watermarked_path: P,
        key: &[u8; 32],
        config: &StenoConfig,
    ) -> Result<Vec<u8>, StenoError> {
        config.validate()?;

        // Read watermarked WAV
        let wav_data = wav::read_wav(&watermarked_path)?;

        // Validate channel selection
        if config.channel >= wav_data.num_channels() {
            return Err(StenoError::Config(format!(
                "channel {} does not exist (file has {} channel{})",
                config.channel,
                wav_data.num_channels(),
                if wav_data.num_channels() == 1 { "" } else { "s" }
            )));
        }

        // Build internal config and decode
        let embed_config = config.to_embed_config(key);
        let samples = &wav_data.channels[config.channel];
        let data = embed::decode_full(samples, &embed_config)?;

        Ok(data)
    }

    /// Encode data into audio samples directly (without file I/O).
    ///
    /// This is for advanced use when you already have audio samples in memory.
    /// The samples are modified in-place.
    ///
    /// # Arguments
    ///
    /// * `samples` — Audio samples as f64 in \[-1.0, 1.0\], modified in place
    /// * `data` — The byte payload to embed
    /// * `key` — 32-byte secret key
    /// * `config` — Encoding parameters
    pub fn encode_samples(
        samples: &mut [f64],
        data: &[u8],
        key: &[u8; 32],
        config: &StenoConfig,
    ) -> Result<(), StenoError> {
        config.validate()?;
        let embed_config = config.to_embed_config(key);
        embed::encode_full(samples, data, &embed_config)?;
        Ok(())
    }

    /// Decode data from audio samples directly (without file I/O).
    ///
    /// This is for advanced use when you already have watermarked audio
    /// samples in memory.
    ///
    /// # Arguments
    ///
    /// * `samples` — Watermarked audio samples as f64
    /// * `key` — 32-byte secret key (must match encoding)
    /// * `config` — Decoding parameters (must match encoding)
    pub fn decode_samples(
        samples: &[f64],
        key: &[u8; 32],
        config: &StenoConfig,
    ) -> Result<Vec<u8>, StenoError> {
        config.validate()?;
        let embed_config = config.to_embed_config(key);
        let data = embed::decode_full(samples, &embed_config)?;
        Ok(data)
    }
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = (i * 7 + 3) as u8;
        }
        key
    }

    // ── Config validation tests ──────────────────────────────────

    #[test]
    fn test_default_config_valid() {
        let config = StenoConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_time_config() {
        let config = StenoConfig::time();
        assert_eq!(config.domain, EmbedDomain::Time);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_frequency_config() {
        let config = StenoConfig::frequency();
        assert_eq!(config.domain, EmbedDomain::Frequency);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_methods() {
        let config = StenoConfig::time()
            .with_dither()
            .with_time_delta(0.01)
            .with_freq_delta(0.1)
            .with_channel(1);

        assert!(config.dither_enabled);
        assert_eq!(config.time_delta, 0.01);
        assert_eq!(config.freq_delta, 0.1);
        assert_eq!(config.channel, 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_time_delta() {
        let config = StenoConfig::default().with_time_delta(0.0);
        assert!(config.validate().is_err());

        let config = StenoConfig::default().with_time_delta(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_freq_delta() {
        let config = StenoConfig::default().with_freq_delta(f64::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_rms_thresholds() {
        let mut config = StenoConfig::default();
        config.rms_low = 0.5;
        config.rms_high = 0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_window_size() {
        let mut config = StenoConfig::default();
        config.freq_window_size = 1000; // Not a power of 2
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_bin_range() {
        let mut config = StenoConfig::default();
        config.freq_min_bin = 500;
        config.freq_max_bin = 100;
        assert!(config.validate().is_err());
    }

    // ── In-memory round-trip tests ───────────────────────────────

    #[test]
    fn test_encode_decode_samples_time() {
        let key = test_key();
        let config = StenoConfig::time();
        let message = b"StenoWav Public API!";

        // Generate a synthetic signal (loud enough for High tier)
        let mut samples: Vec<f64> = (0..200_000)
            .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        Steno::encode_samples(&mut samples, message, &key, &config).unwrap();
        let recovered = Steno::decode_samples(&samples, &key, &config).unwrap();
        assert_eq!(&recovered, message);
    }

    #[test]
    fn test_encode_decode_samples_freq() {
        let key = test_key();
        let config = StenoConfig::frequency();
        let message = b"Freq domain!";

        let mut samples: Vec<f64> = (0..200_000)
            .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        Steno::encode_samples(&mut samples, message, &key, &config).unwrap();
        let recovered = Steno::decode_samples(&samples, &key, &config).unwrap();
        assert_eq!(&recovered, message);
    }

    #[test]
    fn test_encode_decode_samples_with_dither() {
        let key = test_key();
        let config = StenoConfig::time().with_dither();
        let message = b"Dithered!";

        let mut samples: Vec<f64> = (0..200_000)
            .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        Steno::encode_samples(&mut samples, message, &key, &config).unwrap();
        let recovered = Steno::decode_samples(&samples, &key, &config).unwrap();
        assert_eq!(&recovered, message);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key = test_key();
        let config = StenoConfig::time();
        let message = b"Secret";

        let mut samples: Vec<f64> = (0..200_000)
            .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        Steno::encode_samples(&mut samples, message, &key, &config).unwrap();

        // Try decoding with a different key
        let wrong_key = [99u8; 32];
        let result = Steno::decode_samples(&samples, &wrong_key, &config);
        assert!(result.is_err(), "wrong key should fail header validation");
    }

    #[test]
    fn test_invalid_channel() {
        let key = test_key();
        let config = StenoConfig::default().with_channel(5);

        let mut samples: Vec<f64> = vec![0.5; 100_000];
        // encode_samples doesn't check channel (operates directly on samples)
        // but encode/decode with file I/O does — test that via samples API
        // should still work since channel is only used in file API
        Steno::encode_samples(&mut samples, b"test", &key, &config).unwrap();
    }

    // ── File-based round-trip tests ──────────────────────────────

    #[test]
    fn test_encode_decode_file_time_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return; // Skip if test audio not available
        }

        let key = test_key();
        let config = StenoConfig::time();
        let message = b"StenoWav file API test!";

        let output_path = "/tmp/stenowav_test_time.wav";

        Steno::encode(input_path, output_path, message, &key, &config).unwrap();
        let recovered = Steno::decode(output_path, &key, &config).unwrap();
        assert_eq!(&recovered, message);

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_encode_decode_file_freq_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let config = StenoConfig::frequency();
        let message = b"Frequency domain file test!";

        let output_path = "/tmp/stenowav_test_freq.wav";

        Steno::encode(input_path, output_path, message, &key, &config).unwrap();
        let recovered = Steno::decode(output_path, &key, &config).unwrap();
        assert_eq!(&recovered, message);

        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_encode_decode_file_with_dither_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let config = StenoConfig::time().with_dither();
        let message = b"Dithered file API!";

        let output_path = "/tmp/stenowav_test_dither.wav";

        Steno::encode(input_path, output_path, message, &key, &config).unwrap();
        let recovered = Steno::decode(output_path, &key, &config).unwrap();
        assert_eq!(&recovered, message);

        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_wrong_key_file_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let config = StenoConfig::time();

        let output_path = "/tmp/stenowav_test_wrongkey.wav";

        Steno::encode(input_path, output_path, b"Secret", &key, &config).unwrap();

        let wrong_key = [99u8; 32];
        let result = Steno::decode(output_path, &wrong_key, &config);
        assert!(result.is_err());

        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_channel_out_of_bounds() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let config = StenoConfig::default().with_channel(5); // Only 2 channels in stereo

        let result = Steno::encode(input_path, "/tmp/stenowav_oob.wav", b"test", &key, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_various_message_sizes_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let config = StenoConfig::time();

        let messages: &[&[u8]] = &[
            b"A",
            b"Hi",
            b"Hello, StenoWav!",
            b"The quick brown fox jumps over the lazy dog. 1234567890!",
        ];

        for (i, msg) in messages.iter().enumerate() {
            let output_path = format!("/tmp/stenowav_test_size_{}.wav", i);

            Steno::encode(input_path, &output_path, msg, &key, &config).unwrap();
            let recovered = Steno::decode(&output_path, &key, &config).unwrap();
            assert_eq!(
                &recovered, *msg,
                "failed for message of size {}",
                msg.len()
            );

            let _ = std::fs::remove_file(&output_path);
        }
    }

    #[test]
    fn test_delta_sweep_real_audio() {
        let input_path = "../../sounds/MyOwnSummer.wav";
        if !Path::new(input_path).exists() {
            return;
        }

        let key = test_key();
        let message = b"Delta sweep!";

        for delta in &[0.003, 0.005, 0.01, 0.02] {
            let config = StenoConfig::time().with_time_delta(*delta);
            let output_path = format!("/tmp/stenowav_test_delta_{}.wav", delta);

            Steno::encode(input_path, &output_path, message, &key, &config).unwrap();
            let recovered = Steno::decode(&output_path, &key, &config).unwrap();
            assert_eq!(
                &recovered, message,
                "failed at delta={}",
                delta
            );

            let _ = std::fs::remove_file(&output_path);
        }
    }
}
