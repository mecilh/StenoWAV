//! Quantisation Index Modulation (QIM) engine.
//!
//! QIM embeds data by quantising host signal samples to one of two
//! interleaved quantisation lattices, indexed by the bit to embed.
//!
//! ## How it works
//!
//! Given a sample `x`, a step size `delta`, and a bit `b` (0 or 1):
//!
//! 1. Compute the quantisation index: `q = round(x / delta)`
//! 2. If `q mod 2 != b`, shift `q` to the nearest index with correct parity
//! 3. Output the modified sample: `x' = q * delta`
//!
//! Extraction reverses this: `q = round(x' / delta)`, then `b = q mod 2`.
//!
//! The step size `delta` controls the trade-off between robustness
//! (larger delta = more resistant to noise) and imperceptibility
//! (smaller delta = less audible distortion).

use std::fmt;

/// QIM-related errors.
#[derive(Debug, Clone)]
pub enum QimError {
    /// Step size must be positive and non-zero.
    InvalidDelta(f64),
    /// Bit value must be 0 or 1.
    InvalidBit(u8),
}

impl fmt::Display for QimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QimError::InvalidDelta(d) => write!(f, "invalid QIM delta: {} (must be > 0)", d),
            QimError::InvalidBit(b) => write!(f, "invalid bit value: {} (must be 0 or 1)", b),
        }
    }
}

impl std::error::Error for QimError {}

/// QIM configuration.
#[derive(Debug, Clone, Copy)]
pub struct QimConfig {
    /// Quantisation step size. Larger = more robust, less imperceptible.
    /// Must be > 0.
    pub delta: f64,
}

impl QimConfig {
    /// Create a new QIM config with the given step size.
    pub fn new(delta: f64) -> Result<Self, QimError> {
        if delta <= 0.0 || !delta.is_finite() {
            return Err(QimError::InvalidDelta(delta));
        }
        Ok(QimConfig { delta })
    }
}

/// Embed a single bit into a sample using QIM.
///
/// Returns the modified sample value.
///
/// # Arguments
/// * `sample` - The original sample value (f64, typically in [-1.0, 1.0])
/// * `bit` - The bit to embed (0 or 1)
/// * `config` - QIM parameters
pub fn embed_bit(sample: f64, bit: u8, config: &QimConfig) -> Result<f64, QimError> {
    if bit > 1 {
        return Err(QimError::InvalidBit(bit));
    }

    let delta = config.delta;
    let q = (sample / delta).round() as i64;

    // Ensure quantisation index parity matches the bit
    let q_adjusted = if ((q & 1) as u8) != bit {
        // Pick the nearest index with correct parity
        if sample >= q as f64 * delta {
            q + 1
        } else {
            q - 1
        }
    } else {
        q
    };

    Ok(q_adjusted as f64 * delta)
}

/// Extract a single bit from a (potentially modified) sample.
///
/// # Arguments
/// * `sample` - The watermarked sample value
/// * `config` - QIM parameters (must match encoding)
pub fn extract_bit(sample: f64, config: &QimConfig) -> u8 {
    let q = (sample / config.delta).round() as i64;
    (q.rem_euclid(2)) as u8
}

/// Embed multiple bits into consecutive samples.
///
/// # Arguments
/// * `samples` - Mutable slice of samples to modify (must be >= bits.len())
/// * `bits` - Slice of bits to embed (each must be 0 or 1)
/// * `config` - QIM parameters
///
/// Returns the number of bits embedded.
pub fn embed_bits(
    samples: &mut [f64],
    bits: &[u8],
    config: &QimConfig,
) -> Result<usize, QimError> {
    let count = samples.len().min(bits.len());
    for i in 0..count {
        samples[i] = embed_bit(samples[i], bits[i], config)?;
    }
    Ok(count)
}

/// Extract multiple bits from consecutive samples.
///
/// # Arguments
/// * `samples` - Slice of watermarked samples
/// * `config` - QIM parameters (must match encoding)
///
/// Returns a Vec of extracted bits (0 or 1).
pub fn extract_bits(samples: &[f64], config: &QimConfig) -> Vec<u8> {
    samples.iter().map(|&s| extract_bit(s, config)).collect()
}

/// Embed a full byte (8 bits, MSB first) into 8 samples.
///
/// Returns the number of samples modified (always 8 on success).
pub fn embed_byte(
    samples: &mut [f64],
    byte: u8,
    config: &QimConfig,
) -> Result<usize, QimError> {
    assert!(samples.len() >= 8, "need at least 8 samples to embed a byte");

    let bits: Vec<u8> = (0..8).map(|i| (byte >> (7 - i)) & 1).collect();
    embed_bits(samples, &bits, config)
}

/// Extract a full byte (8 bits, MSB first) from 8 samples.
pub fn extract_byte(samples: &[f64], config: &QimConfig) -> u8 {
    assert!(samples.len() >= 8, "need at least 8 samples to extract a byte");

    let bits = extract_bits(&samples[..8], config);
    let mut byte: u8 = 0;
    for (i, &bit) in bits.iter().enumerate() {
        byte |= bit << (7 - i);
    }
    byte
}

/// Compute the maximum distortion introduced by QIM embedding.
///
/// The worst case is delta/2 per sample (when the sample sits exactly
/// between two quantisation levels of the wrong parity).
pub fn max_distortion(config: &QimConfig) -> f64 {
    config.delta
}

/// Compute the Signal-to-Noise Ratio impact of QIM at a given delta.
///
/// For a signal with known RMS, this estimates the SNR in dB
/// after QIM embedding (assuming uniform quantisation noise).
pub fn estimated_snr_db(signal_rms: f64, config: &QimConfig) -> f64 {
    // Quantisation noise power for QIM ~ delta^2 / 4 (worst case parity shift)
    let noise_rms = config.delta / 2.0;
    if noise_rms <= 0.0 || signal_rms <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (signal_rms / noise_rms).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DELTA: f64 = 0.01;

    fn config() -> QimConfig {
        QimConfig::new(DELTA).unwrap()
    }

    #[test]
    fn test_embed_extract_single_bit() {
        let cfg = config();

        // Test embedding and extracting bit 0
        let modified = embed_bit(0.123, 0, &cfg).unwrap();
        assert_eq!(extract_bit(modified, &cfg), 0);

        // Test embedding and extracting bit 1
        let modified = embed_bit(0.123, 1, &cfg).unwrap();
        assert_eq!(extract_bit(modified, &cfg), 1);
    }

    #[test]
    fn test_embed_extract_all_sample_values() {
        let cfg = config();

        // Test across a range of sample values
        for i in -100..=100 {
            let sample = i as f64 / 100.0;
            for bit in 0..=1 {
                let modified = embed_bit(sample, bit, &cfg).unwrap();
                let extracted = extract_bit(modified, &cfg);
                assert_eq!(
                    extracted, bit,
                    "failed at sample={}, bit={}, modified={}",
                    sample, bit, modified
                );
            }
        }
    }

    #[test]
    fn test_distortion_bounded() {
        let cfg = config();
        let max_dist = max_distortion(&cfg);

        for i in -1000..=1000 {
            let sample = i as f64 / 1000.0;
            for bit in 0..=1 {
                let modified = embed_bit(sample, bit, &cfg).unwrap();
                let distortion = (modified - sample).abs();
                assert!(
                    distortion <= max_dist + 1e-12,
                    "distortion {} exceeds max {} at sample={}, bit={}",
                    distortion, max_dist, sample, bit
                );
            }
        }
    }

    #[test]
    fn test_embed_extract_byte() {
        let cfg = config();

        // Test all possible byte values
        for byte_val in 0..=255u8 {
            let mut samples = vec![0.5; 8]; // Arbitrary starting values
            embed_byte(&mut samples, byte_val, &cfg).unwrap();
            let extracted = extract_byte(&samples, &cfg);
            assert_eq!(
                extracted, byte_val,
                "byte round-trip failed for value {}",
                byte_val
            );
        }
    }

    #[test]
    fn test_embed_extract_multi_bits() {
        let cfg = config();
        let bits = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let mut samples = vec![0.3; 8];

        let count = embed_bits(&mut samples, &bits, &cfg).unwrap();
        assert_eq!(count, 8);

        let extracted = extract_bits(&samples, &cfg);
        assert_eq!(extracted, bits);
    }

    #[test]
    fn test_invalid_delta() {
        assert!(QimConfig::new(0.0).is_err());
        assert!(QimConfig::new(-1.0).is_err());
        assert!(QimConfig::new(f64::NAN).is_err());
        assert!(QimConfig::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_invalid_bit() {
        let cfg = config();
        assert!(embed_bit(0.5, 2, &cfg).is_err());
        assert!(embed_bit(0.5, 255, &cfg).is_err());
    }

    #[test]
    fn test_snr_estimate() {
        let cfg = config();
        let snr = estimated_snr_db(0.5, &cfg);
        // For RMS=0.5 and delta=0.01, SNR should be quite high
        assert!(snr > 30.0, "SNR {} should be > 30 dB", snr);
    }
}
