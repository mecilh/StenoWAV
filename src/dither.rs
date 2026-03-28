//! Dither engine and perceptual quality analysis.
//!
//! Dither adds small random perturbations to non-payload samples,
//! raising the noise floor so that QIM-modified samples don't stand
//! out in statistical analysis. The perturbations are constrained to
//! be inaudible (within LSB bounds for 16-bit PCM).
//!
//! ## Design rationale
//!
//! QIM embedding modifies specific samples to align them with a
//! quantisation lattice. An attacker performing statistical analysis
//! could detect that some samples sit on a regular grid while others
//! don't. Dithering spreads small perturbations across ALL non-payload
//! samples, making the entire signal look uniformly "noisy" and hiding
//! the QIM grid pattern.
//!
//! ## Quality metrics
//!
//! The module also provides tools to measure the perceptual impact of
//! watermarking (with or without dither): SNR, max/mean sample
//! distortion, and spectral difference analysis.

use crate::fft;
use crate::packet::{self, PacketError, PAYLOAD_BITS_PER_PACKET};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashSet;

// ══════════════════════════════════════════════════════════════════
// Dither configuration
// ══════════════════════════════════════════════════════════════════

/// Configuration for the dither noise generator.
///
/// Dither is optional and off by default. When enabled, it adds uniform
/// random perturbations to all non-payload samples in eligible signal
/// regions. The perturbation magnitude is bounded by `intensity`.
///
/// ## Audibility constraints
///
/// For 16-bit PCM audio, 1 LSB ≈ 3.05e-5. The default intensity of
/// ~2 LSBs (6.1e-5) is well below audibility thresholds. Even at
/// maximum recommended intensity (~4 LSBs ≈ 1.2e-4), the dither
/// remains inaudible because it's masked by the signal's own
/// quantisation noise.
#[derive(Debug, Clone, Copy)]
pub struct DitherConfig {
    /// Whether dither is enabled. Default: `false`.
    pub enabled: bool,
    /// Maximum perturbation amplitude in normalised sample space [-1, 1].
    /// Each non-payload sample receives a uniform random offset in
    /// `[-intensity, +intensity]`. Default: ~2 LSBs for 16-bit PCM.
    pub intensity: f64,
}

impl Default for DitherConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 2.0 / 32768.0, // ~2 LSBs for 16-bit audio
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// Dither seed derivation
// ══════════════════════════════════════════════════════════════════

/// Derive a dither-specific seed from the master seed.
///
/// Flips byte 2 of the seed to create an independent CSPRNG stream.
/// (Byte 0 is flipped for cipher generation in cipher.rs; byte 2 for dither.)
fn dither_seed(seed: &[u8; 32]) -> [u8; 32] {
    let mut dseed = *seed;
    dseed[2] ^= 0xFF;
    dseed
}

// ══════════════════════════════════════════════════════════════════
// Time-domain dither
// ══════════════════════════════════════════════════════════════════

/// Apply dither to time-domain samples, skipping protected indices.
///
/// Protected indices are those used by the packet system for mask
/// and data embedding. All other samples receive a uniform random
/// perturbation in `[-intensity, +intensity]`.
///
/// Each sample position consumes exactly one RNG value regardless
/// of whether it's protected, ensuring deterministic output for
/// a given seed regardless of which indices are protected.
///
/// Results are clamped to [-1.0, 1.0] to prevent clipping.
pub fn apply_time_dither(
    samples: &mut [f64],
    protected: &HashSet<usize>,
    seed: &[u8; 32],
    config: &DitherConfig,
) {
    if !config.enabled || config.intensity <= 0.0 {
        return;
    }

    let mut rng = ChaCha20Rng::from_seed(dither_seed(seed));

    for i in 0..samples.len() {
        // Always consume one RNG value per sample for determinism
        let noise: f64 = rng.random::<f64>() * 2.0 * config.intensity - config.intensity;

        if !protected.contains(&i) {
            samples[i] = (samples[i] + noise).clamp(-1.0, 1.0);
        }
    }
}

/// Apply dither to frequency-domain magnitudes, skipping protected indices.
///
/// Protected indices are the flat-magnitude-array positions used by
/// the packet system. All other magnitude bins receive a small random
/// perturbation. Magnitudes are clamped to ≥ 0 (negative magnitudes
/// are physically meaningless).
///
/// `intensity` should be scaled appropriately for magnitude values
/// (typically a fraction of the frequency-domain QIM delta).
pub fn apply_freq_dither(
    flat_mags: &mut [f64],
    protected: &HashSet<usize>,
    seed: &[u8; 32],
    intensity: f64,
) {
    if intensity <= 0.0 {
        return;
    }

    let mut rng = ChaCha20Rng::from_seed(dither_seed(seed));

    for i in 0..flat_mags.len() {
        let noise: f64 = rng.random::<f64>() * 2.0 * intensity - intensity;

        if !protected.contains(&i) {
            flat_mags[i] = (flat_mags[i] + noise).max(0.0);
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// Protected index collection
// ══════════════════════════════════════════════════════════════════

/// Collect all sample/bin indices used by the packet system.
///
/// Replays the deterministic index selection for all packets to build
/// the complete set of protected indices. This includes:
/// - 8 mask indices per packet
/// - 16 data indices `[data_index .. data_index + 16)` per packet
///
/// The resulting set tells the dither engine which positions to skip.
pub fn collect_protected_indices(
    seed: &[u8; 32],
    total_packets: usize,
    mask_pool: &[usize],
    data_pool: &[usize],
) -> Result<HashSet<usize>, PacketError> {
    let mut protected = HashSet::new();

    for i in 0..total_packets {
        let indices =
            packet::select_indices_from_pool(seed, i as u64, mask_pool, data_pool)?;

        for &m in &indices.mask_indices {
            protected.insert(m);
        }
        for offset in 0..PAYLOAD_BITS_PER_PACKET {
            protected.insert(indices.data_index + offset);
        }
    }

    Ok(protected)
}

// ══════════════════════════════════════════════════════════════════
// Perceptual quality metrics
// ══════════════════════════════════════════════════════════════════

/// Quality metrics comparing original and watermarked signals.
///
/// All metrics measure the distortion introduced by embedding
/// (and optional dithering). Use [`compute_quality`] to compute.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB: `20·log₁₀(signal_rms / noise_rms)`.
    /// Higher is better. Typical values for imperceptible watermarking: > 30 dB.
    pub snr_db: f64,
    /// Maximum absolute per-sample difference.
    pub max_sample_diff: f64,
    /// Mean absolute per-sample difference.
    pub mean_sample_diff: f64,
    /// RMS of the difference signal (the noise floor).
    pub noise_rms: f64,
    /// RMS of the original signal.
    pub signal_rms: f64,
    /// Number of samples that changed (`|diff| > 1e-15`).
    pub modified_count: usize,
    /// Total number of samples.
    pub total_count: usize,
}

impl std::fmt::Display for QualityMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SNR: {:.1} dB | max Δ: {:.6} | mean Δ: {:.6} | noise RMS: {:.6} | modified: {}/{}",
            self.snr_db,
            self.max_sample_diff,
            self.mean_sample_diff,
            self.noise_rms,
            self.modified_count,
            self.total_count,
        )
    }
}

/// Compute quality metrics between an original and modified signal.
///
/// Both signals must have the same length.
///
/// # Panics
///
/// Panics if `original.len() != modified.len()`.
pub fn compute_quality(original: &[f64], modified: &[f64]) -> QualityMetrics {
    assert_eq!(
        original.len(),
        modified.len(),
        "signal lengths must match: {} vs {}",
        original.len(),
        modified.len()
    );

    let n = original.len();
    if n == 0 {
        return QualityMetrics {
            snr_db: f64::INFINITY,
            max_sample_diff: 0.0,
            mean_sample_diff: 0.0,
            noise_rms: 0.0,
            signal_rms: 0.0,
            modified_count: 0,
            total_count: 0,
        };
    }

    let mut max_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_signal = 0.0f64;
    let mut modified_count = 0usize;

    for i in 0..n {
        let diff = modified[i] - original[i];
        let abs_diff = diff.abs();
        max_diff = max_diff.max(abs_diff);
        sum_abs_diff += abs_diff;
        sum_sq_diff += diff * diff;
        sum_sq_signal += original[i] * original[i];
        if abs_diff > 1e-15 {
            modified_count += 1;
        }
    }

    let noise_rms = (sum_sq_diff / n as f64).sqrt();
    let signal_rms = (sum_sq_signal / n as f64).sqrt();

    let snr_db = if noise_rms > 0.0 {
        20.0 * (signal_rms / noise_rms).log10()
    } else {
        f64::INFINITY
    };

    QualityMetrics {
        snr_db,
        max_sample_diff: max_diff,
        mean_sample_diff: sum_abs_diff / n as f64,
        noise_rms,
        signal_rms,
        modified_count,
        total_count: n,
    }
}

/// Compute the maximum spectral magnitude difference between two signals.
///
/// FFTs both signals (truncated to the largest power-of-2 length that fits)
/// and returns the maximum absolute difference in magnitude across all bins.
///
/// This detects whether watermarking has introduced any concentrated
/// spectral artefacts (e.g., a visible spike in the spectrum).
pub fn max_spectral_diff(original: &[f64], modified: &[f64]) -> Result<f64, fft::FftError> {
    assert_eq!(
        original.len(),
        modified.len(),
        "signal lengths must match"
    );

    let n = original.len();
    if n < 2 {
        return Ok(0.0);
    }

    // Find largest power-of-2 that fits
    let fft_len = 1usize << (usize::BITS - 1 - n.leading_zeros());

    let mut re_orig: Vec<f64> = original[..fft_len].to_vec();
    let mut im_orig = vec![0.0f64; fft_len];
    fft::fft(&mut re_orig, &mut im_orig)?;

    let mut re_mod: Vec<f64> = modified[..fft_len].to_vec();
    let mut im_mod = vec![0.0f64; fft_len];
    fft::fft(&mut re_mod, &mut im_mod)?;

    let mut max_diff = 0.0f64;
    for i in 0..fft_len {
        let mag_orig = fft::complex_mag(re_orig[i], im_orig[i]);
        let mag_mod = fft::complex_mag(re_mod[i], im_mod[i]);
        max_diff = max_diff.max((mag_mod - mag_orig).abs());
    }

    Ok(max_diff)
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seed() -> [u8; 32] {
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = (i * 7 + 3) as u8;
        }
        seed
    }

    // ── DitherConfig defaults ─────────────────────────────────────

    #[test]
    fn test_dither_disabled_by_default() {
        let config = DitherConfig::default();
        assert!(!config.enabled);
        assert!(config.intensity > 0.0);
    }

    // ── Time-domain dither ────────────────────────────────────────

    #[test]
    fn test_dither_disabled_no_change() {
        let mut samples = vec![0.5; 1000];
        let original = samples.clone();
        let config = DitherConfig { enabled: false, intensity: 0.01 };
        let protected = HashSet::new();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);
        assert_eq!(samples, original);
    }

    #[test]
    fn test_dither_zero_intensity_no_change() {
        let mut samples = vec![0.5; 1000];
        let original = samples.clone();
        let config = DitherConfig { enabled: true, intensity: 0.0 };
        let protected = HashSet::new();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);
        assert_eq!(samples, original);
    }

    #[test]
    fn test_dither_applies_perturbations() {
        let mut samples = vec![0.5; 1000];
        let original = samples.clone();
        let config = DitherConfig { enabled: true, intensity: 0.001 };
        let protected = HashSet::new();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);

        // Some samples should have changed
        let changed = samples.iter().zip(&original)
            .filter(|(s, o)| (*s - *o).abs() > 1e-15)
            .count();
        assert!(changed > 900, "expected most samples to change, got {}", changed);
    }

    #[test]
    fn test_dither_respects_intensity_bounds() {
        let mut samples = vec![0.5; 10_000];
        let original = samples.clone();
        let intensity = 0.001;
        let config = DitherConfig { enabled: true, intensity };
        let protected = HashSet::new();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);

        for (i, (s, o)) in samples.iter().zip(&original).enumerate() {
            let diff = (s - o).abs();
            assert!(
                diff <= intensity + 1e-15,
                "sample {} exceeded intensity: diff={}, intensity={}",
                i, diff, intensity
            );
        }
    }

    #[test]
    fn test_dither_skips_protected_indices() {
        let mut samples = vec![0.5; 100];
        let original = samples.clone();
        let config = DitherConfig { enabled: true, intensity: 0.01 };

        // Protect indices 10, 20, 30
        let protected: HashSet<usize> = [10, 20, 30].into_iter().collect();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);

        // Protected samples must be unchanged
        assert_eq!(samples[10], original[10], "protected index 10 was modified");
        assert_eq!(samples[20], original[20], "protected index 20 was modified");
        assert_eq!(samples[30], original[30], "protected index 30 was modified");

        // Non-protected samples should be changed (most of them)
        let changed = samples.iter().zip(&original).enumerate()
            .filter(|(i, _)| !protected.contains(i))
            .filter(|(_, (s, o))| (*s - *o).abs() > 1e-15)
            .count();
        assert!(changed > 80, "expected most non-protected samples to change, got {}", changed);
    }

    #[test]
    fn test_dither_deterministic() {
        let config = DitherConfig { enabled: true, intensity: 0.001 };
        let protected = HashSet::new();
        let seed = test_seed();

        let mut samples1 = vec![0.5; 1000];
        apply_time_dither(&mut samples1, &protected, &seed, &config);

        let mut samples2 = vec![0.5; 1000];
        apply_time_dither(&mut samples2, &protected, &seed, &config);

        assert_eq!(samples1, samples2, "dither should be deterministic");
    }

    #[test]
    fn test_dither_clamps_to_valid_range() {
        // Samples at the edges of the valid range
        let mut samples = vec![0.9999; 500];
        samples.extend(vec![-0.9999; 500]);
        let config = DitherConfig { enabled: true, intensity: 0.01 };
        let protected = HashSet::new();

        apply_time_dither(&mut samples, &protected, &test_seed(), &config);

        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s >= -1.0 && s <= 1.0,
                "sample {} out of range: {}",
                i, s
            );
        }
    }

    // ── Frequency-domain dither ───────────────────────────────────

    #[test]
    fn test_freq_dither_nonnegative() {
        let mut mags = vec![0.001; 500]; // Small but positive
        let protected = HashSet::new();

        apply_freq_dither(&mut mags, &protected, &test_seed(), 0.01);

        for (i, &m) in mags.iter().enumerate() {
            assert!(m >= 0.0, "magnitude {} went negative: {}", i, m);
        }
    }

    #[test]
    fn test_freq_dither_skips_protected() {
        let mut mags = vec![1.0; 100];
        let original = mags.clone();
        let protected: HashSet<usize> = [5, 15, 25].into_iter().collect();

        apply_freq_dither(&mut mags, &protected, &test_seed(), 0.01);

        assert_eq!(mags[5], original[5]);
        assert_eq!(mags[15], original[15]);
        assert_eq!(mags[25], original[25]);
    }

    // ── Quality metrics ──────────────────────────────────────────

    #[test]
    fn test_quality_identical_signals() {
        let signal = vec![0.5; 1000];
        let metrics = compute_quality(&signal, &signal);

        assert_eq!(metrics.snr_db, f64::INFINITY);
        assert_eq!(metrics.max_sample_diff, 0.0);
        assert_eq!(metrics.mean_sample_diff, 0.0);
        assert_eq!(metrics.modified_count, 0);
    }

    #[test]
    fn test_quality_empty_signals() {
        let metrics = compute_quality(&[], &[]);
        assert_eq!(metrics.snr_db, f64::INFINITY);
        assert_eq!(metrics.total_count, 0);
    }

    #[test]
    fn test_quality_known_distortion() {
        let original = vec![1.0; 1000];
        let mut modified = original.clone();
        // Add known distortion to first 100 samples
        for i in 0..100 {
            modified[i] += 0.001;
        }

        let metrics = compute_quality(&original, &modified);

        assert!(metrics.snr_db > 0.0, "SNR should be positive");
        assert!((metrics.max_sample_diff - 0.001).abs() < 1e-10);
        assert_eq!(metrics.modified_count, 100);
        assert_eq!(metrics.total_count, 1000);
    }

    #[test]
    fn test_quality_snr_calculation() {
        // Signal of amplitude 1.0, noise of known magnitude
        let original: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let noise_level = 0.001;
        let modified: Vec<f64> = original.iter().map(|&s| s + noise_level).collect();

        let metrics = compute_quality(&original, &modified);

        // SNR should be roughly 20*log10(signal_rms / noise_level)
        let signal_rms = fft::rms(&original);
        let expected_snr = 20.0 * (signal_rms / noise_level).log10();
        assert!(
            (metrics.snr_db - expected_snr).abs() < 1.0,
            "SNR {:.1} dB too far from expected {:.1} dB",
            metrics.snr_db, expected_snr
        );
    }

    #[test]
    fn test_quality_display() {
        let metrics = QualityMetrics {
            snr_db: 45.3,
            max_sample_diff: 0.005,
            mean_sample_diff: 0.001,
            noise_rms: 0.002,
            signal_rms: 0.5,
            modified_count: 500,
            total_count: 10000,
        };
        let display = format!("{}", metrics);
        assert!(display.contains("45.3 dB"));
        assert!(display.contains("500/10000"));
    }

    // ── Spectral difference ──────────────────────────────────────

    #[test]
    fn test_spectral_diff_identical() {
        let signal: Vec<f64> = (0..1024)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        let diff = max_spectral_diff(&signal, &signal).unwrap();
        assert!(diff < 1e-10, "spectral diff should be ~0, got {}", diff);
    }

    #[test]
    fn test_spectral_diff_with_noise() {
        let signal: Vec<f64> = (0..1024)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut modified = signal.clone();
        // Add small perturbation
        for s in &mut modified {
            *s += 0.001;
        }

        let diff = max_spectral_diff(&signal, &modified).unwrap();
        assert!(diff > 0.0, "spectral diff should be > 0");
        // The perturbation is tiny, so spectral diff should be bounded
        assert!(diff < 5.0, "spectral diff should be small, got {}", diff);
    }

    // ── Protected index collection ───────────────────────────────

    #[test]
    fn test_collect_protected_indices() {
        // Build a simple pool (0..10000 as both mask and data pool)
        let pool: Vec<usize> = (0..10_000).collect();

        let protected = collect_protected_indices(
            &test_seed(), 3, &pool, &pool,
        ).unwrap();

        // 3 packets × (8 mask + 16 data) = at most 72 indices
        // Could be fewer if there's overlap between packets
        assert!(protected.len() <= 72);
        assert!(protected.len() >= 24, "expected at least 24 protected, got {}", protected.len());
    }
}
