//! Dual-domain embedding engine with RMS-adaptive bit placement.
//!
//! This is the high-level API for encoding and decoding data in audio signals.
//! It wraps the packet system with RMS-adaptive sample selection and supports
//! both time-domain (direct PCM) and frequency-domain (FFT magnitude) embedding.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────┐       ┌─────────────┐
//!  │ Time-domain  │       │ Freq-domain  │
//!  │   samples    │       │  magnitudes  │
//!  └──────┬───────┘       └──────┬───────┘
//!         │                      │
//!    ┌────▼────────────────────▼────┐
//!    │   RMS segment analysis        │
//!    │   → Power tier classification  │
//!    │   → Eligible index pools       │
//!    └────┬─────────────────────────┘
//!         │
//!    ┌────▼─────────┐
//!    │ Packet system │  (mask/cipher + data embedding via QIM)
//!    └──────────────┘
//! ```
//!
//! ## Parameter sweep support
//!
//! All thresholds, deltas, segment sizes, and bit placement rules are exposed
//! as public struct fields. This allows simulation sweeps (e.g. AWGN loss
//! function analysis across 1000s of samples) without touching any logic.

use crate::cipher::{self, CipherKey};
use crate::dither::{self, DitherConfig};
use crate::fft;
use crate::packet::{
    self, PacketError, MASK_SAMPLE_COUNT, PAYLOAD_BITS_PER_PACKET, PAYLOAD_BYTES_PER_PACKET,
};
use crate::qim::{QimConfig, QimError};
use std::fmt;

// ══════════════════════════════════════════════════════════════════
// Configuration types — all fields public for simulation sweeps
// ══════════════════════════════════════════════════════════════════

/// Power tier classification for a signal segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerTier {
    /// High RMS — allow embedding in higher-significance bits.
    High,
    /// Mid RMS — restrict to lower-significance bits.
    Mid,
    /// Low RMS — skip entirely (no embedding here).
    Skip,
}

/// RMS thresholds for power tier classification.
///
/// - RMS >= `high` → [`PowerTier::High`]
/// - `low` <= RMS < `high` → [`PowerTier::Mid`]
/// - RMS < `low` → [`PowerTier::Skip`]
///
/// Thresholds are on normalised PCM values (signal in \[-1, 1\]).
/// Set these with margin above QIM delta to ensure stable classification
/// across encode/decode (QIM distortion shifts RMS by at most ~delta).
#[derive(Debug, Clone, Copy)]
pub struct RmsThresholds {
    /// Boundary between Mid and High tiers.
    pub high: f64,
    /// Boundary between Skip and Mid tiers.
    pub low: f64,
}

impl Default for RmsThresholds {
    fn default() -> Self {
        Self {
            high: 0.10,
            low: 0.01,
        }
    }
}

/// Bit placement rules per power tier.
///
/// Controls how many bits per sample are embedded in each tier.
/// Higher values = more data density but more distortion.
/// Currently used for documentation/future use — the packet system
/// uses 1 bit per data sample and 2 bits per mask sample (mod-4 QIM).
#[derive(Debug, Clone, Copy)]
pub struct BitPlacementRules {
    /// Bits per sample in high-power segments.
    pub high_tier_bits: u8,
    /// Bits per sample in mid-power segments.
    pub mid_tier_bits: u8,
}

impl Default for BitPlacementRules {
    fn default() -> Self {
        Self {
            high_tier_bits: 1,
            mid_tier_bits: 1,
        }
    }
}

/// Which domain(s) to embed data in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedDomain {
    /// Embed directly in PCM samples (time-domain QIM).
    Time,
    /// Embed in FFT magnitude coefficients (frequency-domain QIM).
    Frequency,
}

/// Frequency-domain specific parameters.
///
/// These control the FFT analysis/synthesis process and which frequency
/// bins are eligible for embedding.
#[derive(Debug, Clone, Copy)]
pub struct FreqConfig {
    /// FFT window size (must be power of 2). Larger windows give finer
    /// frequency resolution but fewer frames (fewer packets).
    pub window_size: usize,
    /// Lowest frequency bin index to embed in. Skip DC (bin 0) and very
    /// low frequencies which carry perceptually important content.
    pub min_bin: usize,
    /// Highest frequency bin index to embed in (exclusive). Should be
    /// ≤ window_size / 2 (Nyquist). Bins above this are mirrored.
    pub max_bin: usize,
    /// QIM delta for frequency-domain magnitudes. Frequency magnitudes
    /// can be much larger than PCM samples, so this delta can differ
    /// from the time-domain delta.
    pub delta: f64,
}

impl Default for FreqConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            min_bin: 10,
            max_bin: 400,
            delta: 0.05,
        }
    }
}

/// RMS analysis parameters.
#[derive(Debug, Clone, Copy)]
pub struct RmsConfig {
    /// Number of samples per segment for RMS computation.
    /// Smaller segments = finer-grained power classification.
    pub segment_size: usize,
    /// Power tier thresholds.
    pub thresholds: RmsThresholds,
}

impl Default for RmsConfig {
    fn default() -> Self {
        Self {
            segment_size: 4096,
            thresholds: RmsThresholds::default(),
        }
    }
}

/// Top-level embedding configuration.
///
/// Every field is public and tunable for simulation sweeps.
/// Use [`EmbedConfig::default()`] for sane defaults that work out of the box.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Which domain to embed in.
    pub domain: EmbedDomain,
    /// QIM delta for time-domain embedding (PCM samples in \[-1, 1\]).
    pub time_delta: f64,
    /// RMS analysis and classification parameters.
    pub rms: RmsConfig,
    /// Bit placement rules per power tier.
    pub bit_placement: BitPlacementRules,
    /// Frequency-domain parameters (only used when domain is Frequency).
    pub freq: FreqConfig,
    /// 32-byte seed for the CSPRNG (index selection + cipher generation).
    pub seed: [u8; 32],
    /// Dither configuration (optional noise to mask QIM patterns).
    pub dither: DitherConfig,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            domain: EmbedDomain::Time,
            time_delta: 0.005,
            rms: RmsConfig::default(),
            bit_placement: BitPlacementRules::default(),
            freq: FreqConfig::default(),
            seed: [0u8; 32],
            dither: DitherConfig::default(),
        }
    }
}

/// Metadata returned from encoding, required for decoding.
#[derive(Debug, Clone)]
pub struct EncodeMeta {
    /// Number of packets encoded.
    pub num_packets: usize,
    /// Original data length in bytes (before padding).
    pub original_len: usize,
    /// Domain used for encoding (must match during decoding).
    pub domain: EmbedDomain,
}

// ══════════════════════════════════════════════════════════════════
// Errors
// ══════════════════════════════════════════════════════════════════

/// Errors from the embedding engine.
#[derive(Debug, Clone)]
pub enum EmbedError {
    /// Not enough eligible samples or bins to embed the data.
    InsufficientCapacity(String),
    /// QIM configuration error.
    Qim(QimError),
    /// Packet-level error.
    Packet(PacketError),
    /// FFT error.
    Fft(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::InsufficientCapacity(msg) => write!(f, "insufficient capacity: {}", msg),
            EmbedError::Qim(e) => write!(f, "QIM error: {}", e),
            EmbedError::Packet(e) => write!(f, "packet error: {}", e),
            EmbedError::Fft(msg) => write!(f, "FFT error: {}", msg),
            EmbedError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<PacketError> for EmbedError {
    fn from(e: PacketError) -> Self {
        EmbedError::Packet(e)
    }
}

impl From<QimError> for EmbedError {
    fn from(e: QimError) -> Self {
        EmbedError::Qim(e)
    }
}

impl From<fft::FftError> for EmbedError {
    fn from(e: fft::FftError) -> Self {
        EmbedError::Fft(e.to_string())
    }
}

// ══════════════════════════════════════════════════════════════════
// RMS segment analysis
// ══════════════════════════════════════════════════════════════════

/// Analysis result for a single signal segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Start index in the sample array.
    pub start: usize,
    /// Number of samples in this segment.
    pub len: usize,
    /// Computed RMS value.
    pub rms: f64,
    /// Power tier classification.
    pub tier: PowerTier,
}

/// Analyse signal segments and classify each by RMS power tier.
///
/// Divides `samples` into consecutive segments of `config.segment_size`
/// (last segment may be shorter) and computes the RMS of each.
pub fn analyze_segments(samples: &[f64], config: &RmsConfig) -> Vec<SegmentInfo> {
    let mut segments = Vec::new();
    let mut start = 0;

    while start < samples.len() {
        let end = (start + config.segment_size).min(samples.len());
        let rms_val = fft::rms(&samples[start..end]);
        let tier = classify_rms(rms_val, &config.thresholds);

        segments.push(SegmentInfo {
            start,
            len: end - start,
            rms: rms_val,
            tier,
        });
        start = end;
    }

    segments
}

/// Classify an RMS value into a power tier.
fn classify_rms(rms: f64, thresholds: &RmsThresholds) -> PowerTier {
    if rms >= thresholds.high {
        PowerTier::High
    } else if rms >= thresholds.low {
        PowerTier::Mid
    } else {
        PowerTier::Skip
    }
}

/// Build pools of eligible sample indices from segment analysis.
///
/// Returns `(mask_pool, data_pool)`:
/// - `mask_pool`: all individual indices in High or Mid segments.
/// - `data_pool`: indices where a full 16-sample data region fits
///   within an eligible segment without exceeding the signal.
fn build_eligible_pools(segments: &[SegmentInfo], signal_len: usize) -> (Vec<usize>, Vec<usize>) {
    let mut mask_pool = Vec::new();
    let mut data_pool = Vec::new();

    for seg in segments {
        if seg.tier == PowerTier::Skip {
            continue;
        }

        let seg_end = (seg.start + seg.len).min(signal_len);

        // Every index in an eligible segment is a valid mask position
        for i in seg.start..seg_end {
            mask_pool.push(i);
        }

        // A valid data-start index needs 16 contiguous samples within the segment
        if seg_end.saturating_sub(seg.start) >= PAYLOAD_BITS_PER_PACKET {
            let max_data_start = seg_end - PAYLOAD_BITS_PER_PACKET;
            for i in seg.start..=max_data_start {
                data_pool.push(i);
            }
        }
    }

    (mask_pool, data_pool)
}

// ══════════════════════════════════════════════════════════════════
// Time-domain embedding
// ══════════════════════════════════════════════════════════════════

/// Encode data into a signal using time-domain QIM with RMS-adaptive placement.
///
/// Analyses the signal's RMS per segment to classify power tiers, then
/// restricts packet index selection to High and Mid regions only.
pub fn encode_time(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    let qim_config = QimConfig::new(config.time_delta)?;
    let segments = analyze_segments(samples, &config.rms);
    let (mask_pool, data_pool) = build_eligible_pools(&segments, samples.len());

    validate_pool_capacity(&mask_pool, &data_pool)?;

    let (padded, num_packets) = pad_data(data);

    for i in 0..num_packets {
        let offset = i * PAYLOAD_BYTES_PER_PACKET;
        let payload = [padded[offset], padded[offset + 1]];

        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        packet::encode_packet_at(
            samples,
            &config.seed,
            i as u64,
            payload,
            &qim_config,
            &indices,
        )?;
    }

    Ok(EncodeMeta {
        num_packets,
        original_len: data.len(),
        domain: EmbedDomain::Time,
    })
}

/// Decode data from a time-domain watermarked signal.
///
/// Recomputes the same RMS analysis and index pools to locate embedded packets.
pub fn decode_time(
    samples: &[f64],
    meta: &EncodeMeta,
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    let qim_config = QimConfig::new(config.time_delta)?;
    let segments = analyze_segments(samples, &config.rms);
    let (mask_pool, data_pool) = build_eligible_pools(&segments, samples.len());

    let mut output = Vec::with_capacity(meta.num_packets * PAYLOAD_BYTES_PER_PACKET);

    for i in 0..meta.num_packets {
        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        let payload = packet::decode_packet_at(
            samples,
            &config.seed,
            i as u64,
            &qim_config,
            &indices,
        )?;

        output.extend_from_slice(&payload);
    }

    output.truncate(meta.original_len);
    Ok(output)
}

// ══════════════════════════════════════════════════════════════════
// Non-negative QIM for frequency-domain magnitudes
// ══════════════════════════════════════════════════════════════════
//
// Magnitudes (sqrt(re²+im²)) are always ≥ 0. Standard QIM can push
// values negative, but negative magnitudes lose their sign through
// the FFT→IFFT→FFT round-trip. The cipher module provides non-negative
// variants that constrain quantisation indices to ≥ 0.
//
// Single-bit non-negative QIM wrappers are kept here since the cipher
// module focuses on mod-4 (2-bit) operations for the mask samples.

/// Single-bit non-negative QIM: embed a bit into a magnitude.
fn embed_bit_nonneg(mag: f64, bit: u8, delta: f64) -> f64 {
    // Use the cipher module's non-negative symbol embed with m=2
    cipher::embed_symbol_nonneg(mag, bit, 2, delta)
}

/// Single-bit non-negative QIM: extract a bit from a magnitude.
fn extract_bit_nonneg(mag: f64, delta: f64) -> u8 {
    cipher::extract_symbol_nonneg(mag, 2, delta)
}

// ══════════════════════════════════════════════════════════════════
// Frequency-domain embedding
// ══════════════════════════════════════════════════════════════════

/// Internal state from FFT analysis of the signal.
struct FreqAnalysis {
    /// Per-frame magnitude spectra: magnitudes[frame][bin] for bins 0..N/2+1.
    magnitudes: Vec<Vec<f64>>,
    /// Per-frame phase spectra: phases[frame][bin] for bins 0..N/2+1.
    phases: Vec<Vec<f64>>,
    /// Number of complete frames.
    num_frames: usize,
    /// Length of signal covered by complete frames (num_frames * window_size).
    used_len: usize,
}

/// Analyse the signal in the frequency domain: split into non-overlapping
/// frames, FFT each, return magnitudes and phases.
///
/// Only processes **complete** frames (tail samples shorter than window_size
/// are skipped). This ensures the FFT→IFFT round-trip is exact — no
/// zero-padding artifacts that would corrupt extraction.
fn freq_analyze(samples: &[f64], freq: &FreqConfig) -> Result<FreqAnalysis, EmbedError> {
    let ws = freq.window_size;
    if ws == 0 || (ws & (ws - 1)) != 0 {
        return Err(EmbedError::InvalidConfig(format!(
            "window_size {} must be a power of 2",
            ws
        )));
    }
    if freq.min_bin >= freq.max_bin || freq.max_bin > ws / 2 + 1 {
        return Err(EmbedError::InvalidConfig(format!(
            "invalid bin range [{}, {}), max allowed = {}",
            freq.min_bin,
            freq.max_bin,
            ws / 2 + 1
        )));
    }

    let num_frames = samples.len() / ws; // Only complete frames
    let used_len = num_frames * ws;
    let half = ws / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_frames);
    let mut phases = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let frame_start = f * ws;
        let mut re: Vec<f64> = samples[frame_start..frame_start + ws].to_vec();
        let mut im = vec![0.0f64; ws];

        fft::fft(&mut re, &mut im)?;

        let mut mag = Vec::with_capacity(half);
        let mut phase = Vec::with_capacity(half);
        for b in 0..half {
            mag.push(fft::complex_mag(re[b], im[b]));
            phase.push(fft::complex_phase(re[b], im[b]));
        }

        magnitudes.push(mag);
        phases.push(phase);
    }

    Ok(FreqAnalysis {
        magnitudes,
        phases,
        num_frames,
        used_len,
    })
}

/// Synthesise a time-domain signal from (potentially modified) magnitudes
/// and original phases. Mirrors conjugate bins and applies IFFT per frame.
///
/// Returns a vector of length `used_len` (complete frames only).
fn freq_synthesize(
    analysis: &FreqAnalysis,
    freq: &FreqConfig,
) -> Result<Vec<f64>, EmbedError> {
    let ws = freq.window_size;
    let half = ws / 2 + 1;
    let mut output = vec![0.0f64; analysis.used_len];

    for f in 0..analysis.num_frames {
        let mag = &analysis.magnitudes[f];
        let phase = &analysis.phases[f];

        let mut re = vec![0.0f64; ws];
        let mut im = vec![0.0f64; ws];

        // Reconstruct complex spectrum from magnitude + phase
        for b in 0..half {
            re[b] = mag[b] * phase[b].cos();
            im[b] = mag[b] * phase[b].sin();
        }

        // Mirror conjugate bins: bin[N-k] = conj(bin[k]) for k=1..N/2-1
        for b in 1..(ws / 2) {
            re[ws - b] = re[b];
            im[ws - b] = -im[b];
        }

        fft::ifft(&mut re, &mut im)?;

        let frame_start = f * ws;
        for i in 0..ws {
            output[frame_start + i] = re[i];
        }
    }

    Ok(output)
}

/// Build a flat magnitude array from eligible frames and the index mapping.
///
/// Returns `(flat_mags, mask_pool, data_pool, frame_bin_map)`.
fn build_freq_pools(
    analysis: &FreqAnalysis,
    rms_config: &RmsConfig,
    freq: &FreqConfig,
    time_samples: &[f64],
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<(usize, usize)>) {
    let ws = freq.window_size;
    let bins_per_frame = freq.max_bin - freq.min_bin;

    // Classify each frame by time-domain RMS
    let frame_tiers: Vec<PowerTier> = (0..analysis.num_frames)
        .map(|f| {
            let start = f * ws;
            let end = (start + ws).min(time_samples.len());
            if start >= time_samples.len() {
                return PowerTier::Skip;
            }
            let rms_val = fft::rms(&time_samples[start..end]);
            classify_rms(rms_val, &rms_config.thresholds)
        })
        .collect();

    let mut flat_mags = Vec::new();
    let mut mask_pool = Vec::new();
    let mut data_pool = Vec::new();
    let mut frame_bin_map = Vec::new();

    for (f, &tier) in frame_tiers.iter().enumerate() {
        if tier == PowerTier::Skip {
            continue;
        }

        let base_flat = flat_mags.len();

        for bin in freq.min_bin..freq.max_bin {
            mask_pool.push(flat_mags.len());
            frame_bin_map.push((f, bin));
            flat_mags.push(analysis.magnitudes[f][bin]);
        }

        if bins_per_frame >= PAYLOAD_BITS_PER_PACKET {
            for local in 0..=(bins_per_frame - PAYLOAD_BITS_PER_PACKET) {
                data_pool.push(base_flat + local);
            }
        }
    }

    (flat_mags, mask_pool, data_pool, frame_bin_map)
}

/// Embed cipher into mask positions using non-negative mod-4 QIM.
/// Delegates to the cipher module's magnitude-aware embedding.
fn freq_embed_cipher(
    flat_mags: &mut [f64],
    mask_indices: &[usize; MASK_SAMPLE_COUNT],
    cipher: CipherKey,
    delta: f64,
) {
    cipher::embed_into_magnitudes(flat_mags, mask_indices, cipher, delta);
}

/// Extract cipher from mask positions using non-negative mod-4 QIM.
/// Delegates to the cipher module's magnitude-aware extraction.
fn freq_extract_cipher(
    flat_mags: &[f64],
    mask_indices: &[usize; MASK_SAMPLE_COUNT],
    delta: f64,
) -> CipherKey {
    cipher::extract_from_magnitudes(flat_mags, mask_indices, delta)
}

/// Embed payload into data region using non-negative single-bit QIM.
fn freq_embed_payload(
    flat_mags: &mut [f64],
    data_index: usize,
    payload: [u8; PAYLOAD_BYTES_PER_PACKET],
    cipher: CipherKey,
    delta: f64,
) {
    let payload_u16 = u16::from_be_bytes(payload);
    let masked = cipher.apply(payload_u16);
    for bit_pos in 0..PAYLOAD_BITS_PER_PACKET {
        let bit = ((masked >> (PAYLOAD_BITS_PER_PACKET - 1 - bit_pos)) & 1) as u8;
        let idx = data_index + bit_pos;
        flat_mags[idx] = embed_bit_nonneg(flat_mags[idx], bit, delta);
    }
}

/// Extract payload from data region using non-negative single-bit QIM.
fn freq_extract_payload(
    flat_mags: &[f64],
    data_index: usize,
    cipher: CipherKey,
    delta: f64,
) -> [u8; PAYLOAD_BYTES_PER_PACKET] {
    let mut masked: u16 = 0;
    for bit_pos in 0..PAYLOAD_BITS_PER_PACKET {
        let idx = data_index + bit_pos;
        let bit = extract_bit_nonneg(flat_mags[idx], delta);
        masked |= (bit as u16) << (PAYLOAD_BITS_PER_PACKET - 1 - bit_pos);
    }
    let payload_u16 = cipher.apply(masked);
    payload_u16.to_be_bytes()
}

/// Encode data into a signal using frequency-domain QIM.
///
/// Splits the signal into non-overlapping FFT frames, embeds data in
/// selected magnitude bins using non-negative QIM, then reconstructs
/// the time-domain signal preserving original phases.
pub fn encode_freq(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    let freq = &config.freq;
    if freq.delta <= 0.0 || !freq.delta.is_finite() {
        return Err(EmbedError::InvalidConfig(format!(
            "freq delta {} must be positive and finite",
            freq.delta
        )));
    }

    let mut analysis = freq_analyze(samples, freq)?;
    let (mut flat_mags, mask_pool, data_pool, frame_bin_map) =
        build_freq_pools(&analysis, &config.rms, freq, samples);

    validate_pool_capacity(&mask_pool, &data_pool)?;

    let (padded, num_packets) = pad_data(data);

    for i in 0..num_packets {
        let offset = i * PAYLOAD_BYTES_PER_PACKET;
        let payload = [padded[offset], padded[offset + 1]];

        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        // Use non-negative QIM for magnitude embedding
        let cipher = packet::generate_cipher(&config.seed, i as u64);
        freq_embed_cipher(&mut flat_mags, &indices.mask_indices, cipher, freq.delta);
        freq_embed_payload(&mut flat_mags, indices.data_index, payload, cipher, freq.delta);
    }

    // Scatter modified magnitudes back into per-frame arrays
    for (flat_idx, &mag) in flat_mags.iter().enumerate() {
        let (frame, bin) = frame_bin_map[flat_idx];
        analysis.magnitudes[frame][bin] = mag;
    }

    // Synthesise modified time-domain signal (only overwrites complete frames)
    let modified = freq_synthesize(&analysis, freq)?;
    samples[..analysis.used_len].copy_from_slice(&modified);

    Ok(EncodeMeta {
        num_packets,
        original_len: data.len(),
        domain: EmbedDomain::Frequency,
    })
}

/// Decode data from a frequency-domain watermarked signal.
///
/// Re-analyses the watermarked signal with the same FFT parameters,
/// rebuilds the same index pools, and extracts embedded data from
/// the magnitude spectrum using non-negative QIM.
pub fn decode_freq(
    samples: &[f64],
    meta: &EncodeMeta,
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    let freq = &config.freq;

    let analysis = freq_analyze(samples, freq)?;
    let (flat_mags, mask_pool, data_pool, _frame_bin_map) =
        build_freq_pools(&analysis, &config.rms, freq, samples);

    let mut output = Vec::with_capacity(meta.num_packets * PAYLOAD_BYTES_PER_PACKET);

    for i in 0..meta.num_packets {
        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        let cipher = freq_extract_cipher(&flat_mags, &indices.mask_indices, freq.delta);
        let payload = freq_extract_payload(&flat_mags, indices.data_index, cipher, freq.delta);
        output.extend_from_slice(&payload);
    }

    output.truncate(meta.original_len);
    Ok(output)
}

// ══════════════════════════════════════════════════════════════════
// Top-level API — dispatches to the appropriate domain
// ══════════════════════════════════════════════════════════════════

/// Encode arbitrary data into an audio signal.
///
/// Dispatches to [`encode_time`] or [`encode_freq`] based on `config.domain`.
/// Returns metadata needed for decoding.
pub fn encode(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    match config.domain {
        EmbedDomain::Time => encode_time(samples, data, config),
        EmbedDomain::Frequency => encode_freq(samples, data, config),
    }
}

/// Decode data from a watermarked audio signal.
///
/// Dispatches to [`decode_time`] or [`decode_freq`] based on `config.domain`.
/// The `meta` parameter must match the encoding session.
pub fn decode(
    samples: &[f64],
    meta: &EncodeMeta,
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    match config.domain {
        EmbedDomain::Time => decode_time(samples, meta, config),
        EmbedDomain::Frequency => decode_freq(samples, meta, config),
    }
}

// ══════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════

/// Pad data to an even number of bytes and compute packet count.
fn pad_data(data: &[u8]) -> (Vec<u8>, usize) {
    let mut padded = data.to_vec();
    if padded.len() % PAYLOAD_BYTES_PER_PACKET != 0 {
        padded.push(0x00);
    }
    let num_packets = padded.len() / PAYLOAD_BYTES_PER_PACKET;
    (padded, num_packets)
}

/// Validate that pools have enough capacity for at least one packet.
fn validate_pool_capacity(mask_pool: &[usize], data_pool: &[usize]) -> Result<(), EmbedError> {
    if mask_pool.len() < MASK_SAMPLE_COUNT {
        return Err(EmbedError::InsufficientCapacity(format!(
            "only {} eligible mask positions, need at least {}",
            mask_pool.len(),
            MASK_SAMPLE_COUNT
        )));
    }
    if data_pool.is_empty() {
        return Err(EmbedError::InsufficientCapacity(
            "no eligible data regions (segments too short or all below RMS threshold)".into(),
        ));
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════
// Header — self-contained metadata embedding
// ══════════════════════════════════════════════════════════════════
//
// The header embeds encoding metadata (magic, domain, data length,
// packet count) into the first packets of the watermark. This allows
// the decoder to operate with just the watermarked signal + seed —
// no out-of-band EncodeMeta required.
//
// Header format (14 bytes = 7 packets at 2 bytes each):
//   bytes  0–3:  magic "STEN" (0x5354454E)
//   byte   4:    domain (0 = Time, 1 = Frequency)
//   byte   5:    reserved (0x00)
//   bytes  6–9:  original_len as u32 big-endian
//   bytes 10–13: num_data_packets as u32 big-endian

/// Magic bytes identifying a StenoWav header.
const HEADER_MAGIC: [u8; 4] = [0x53, 0x54, 0x45, 0x4E]; // "STEN"

/// Total header size in bytes.
const HEADER_SIZE: usize = 14;

/// Number of packets consumed by the header.
const HEADER_PACKETS: usize = (HEADER_SIZE + PAYLOAD_BYTES_PER_PACKET - 1) / PAYLOAD_BYTES_PER_PACKET; // 7

/// Error returned when the header cannot be decoded.
#[derive(Debug, Clone)]
pub struct HeaderError(pub String);

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "header error: {}", self.0)
    }
}

impl std::error::Error for HeaderError {}

/// Decoded header contents.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Header {
    domain: EmbedDomain,
    original_len: usize,
    num_data_packets: usize,
}

/// Serialize a header into bytes.
fn serialize_header(domain: EmbedDomain, original_len: usize, num_data_packets: usize) -> [u8; HEADER_SIZE] {
    let mut buf = [0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(&HEADER_MAGIC);
    buf[4] = match domain {
        EmbedDomain::Time => 0,
        EmbedDomain::Frequency => 1,
    };
    buf[5] = 0x00; // reserved
    buf[6..10].copy_from_slice(&(original_len as u32).to_be_bytes());
    buf[10..14].copy_from_slice(&(num_data_packets as u32).to_be_bytes());
    buf
}

/// Deserialize a header from bytes.
fn deserialize_header(buf: &[u8; HEADER_SIZE]) -> Result<Header, HeaderError> {
    if buf[0..4] != HEADER_MAGIC {
        return Err(HeaderError(format!(
            "invalid magic: expected STEN, got [{:#04X}, {:#04X}, {:#04X}, {:#04X}]",
            buf[0], buf[1], buf[2], buf[3]
        )));
    }

    let domain = match buf[4] {
        0 => EmbedDomain::Time,
        1 => EmbedDomain::Frequency,
        other => return Err(HeaderError(format!("unknown domain byte: {}", other))),
    };

    let original_len = u32::from_be_bytes([buf[6], buf[7], buf[8], buf[9]]) as usize;
    let num_data_packets = u32::from_be_bytes([buf[10], buf[11], buf[12], buf[13]]) as usize;

    Ok(Header {
        domain,
        original_len,
        num_data_packets,
    })
}

// ══════════════════════════════════════════════════════════════════
// Full encode/decode pipeline — self-contained with embedded header
// ══════════════════════════════════════════════════════════════════

/// Encode data with an embedded header for self-contained decoding.
///
/// This is the full encoder pipeline: it embeds a header (magic, domain,
/// data length, packet count) followed by the data packets. The decoder
/// needs only the watermarked signal, seed, and config — no `EncodeMeta`.
///
/// Packet layout:
/// ```text
///   packets 0..6:  header (14 bytes of metadata)
///   packets 7..N:  data payload
/// ```
///
/// Returns `EncodeMeta` for compatibility, but the decoder doesn't need it.
pub fn encode_full(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    match config.domain {
        EmbedDomain::Time => encode_full_time(samples, data, config),
        EmbedDomain::Frequency => encode_full_freq(samples, data, config),
    }
}

/// Decode data using only the watermarked signal, seed, and config.
///
/// Extracts the embedded header to learn the domain, data length, and
/// packet count, then extracts the data packets. No `EncodeMeta` needed.
///
/// Returns `Err` if the header magic doesn't match (wrong seed, wrong
/// config, or the signal was never watermarked).
pub fn decode_full(
    samples: &[f64],
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    match config.domain {
        EmbedDomain::Time => decode_full_time(samples, config),
        EmbedDomain::Frequency => decode_full_freq(samples, config),
    }
}

// ── Time-domain full pipeline ────────────────────────────────────

fn encode_full_time(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    let qim_config = QimConfig::new(config.time_delta)?;
    let segments = analyze_segments(samples, &config.rms);
    let (mask_pool, data_pool) = build_eligible_pools(&segments, samples.len());
    validate_pool_capacity(&mask_pool, &data_pool)?;

    // Compute data packets
    let (padded_data, num_data_packets) = pad_data(data);

    // Build header
    let header_bytes = serialize_header(EmbedDomain::Time, data.len(), num_data_packets);

    // Combine: header + data
    let total_payload = build_combined_payload(&header_bytes, &padded_data);
    let total_packets = HEADER_PACKETS + num_data_packets;

    // Encode all packets (header first, then data)
    for i in 0..total_packets {
        let offset = i * PAYLOAD_BYTES_PER_PACKET;
        let payload = [total_payload[offset], total_payload[offset + 1]];

        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        packet::encode_packet_at(
            samples,
            &config.seed,
            i as u64,
            payload,
            &qim_config,
            &indices,
        )?;
    }

    // Apply dither to non-payload samples (raises noise floor to mask QIM patterns)
    if config.dither.enabled {
        let protected = dither::collect_protected_indices(
            &config.seed, total_packets, &mask_pool, &data_pool,
        )?;
        dither::apply_time_dither(samples, &protected, &config.seed, &config.dither);
    }

    Ok(EncodeMeta {
        num_packets: total_packets,
        original_len: data.len(),
        domain: EmbedDomain::Time,
    })
}

fn decode_full_time(
    samples: &[f64],
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    let qim_config = QimConfig::new(config.time_delta)?;
    let segments = analyze_segments(samples, &config.rms);
    let (mask_pool, data_pool) = build_eligible_pools(&segments, samples.len());

    // Extract header packets first
    let header = extract_header(samples, config, &qim_config, &mask_pool, &data_pool)?;

    // Extract data packets
    let mut output = Vec::with_capacity(header.num_data_packets * PAYLOAD_BYTES_PER_PACKET);
    for i in 0..header.num_data_packets {
        let packet_num = (HEADER_PACKETS + i) as u64;
        let indices =
            packet::select_indices_from_pool(&config.seed, packet_num, &mask_pool, &data_pool)?;

        let payload = packet::decode_packet_at(
            samples,
            &config.seed,
            packet_num,
            &qim_config,
            &indices,
        )?;

        output.extend_from_slice(&payload);
    }

    output.truncate(header.original_len);
    Ok(output)
}

// ── Frequency-domain full pipeline ───────────────────────────────

fn encode_full_freq(
    samples: &mut [f64],
    data: &[u8],
    config: &EmbedConfig,
) -> Result<EncodeMeta, EmbedError> {
    let freq = &config.freq;
    if freq.delta <= 0.0 || !freq.delta.is_finite() {
        return Err(EmbedError::InvalidConfig(format!(
            "freq delta {} must be positive and finite",
            freq.delta
        )));
    }

    let mut analysis = freq_analyze(samples, freq)?;
    let (mut flat_mags, mask_pool, data_pool, frame_bin_map) =
        build_freq_pools(&analysis, &config.rms, freq, samples);
    validate_pool_capacity(&mask_pool, &data_pool)?;

    let (padded_data, num_data_packets) = pad_data(data);
    let header_bytes = serialize_header(EmbedDomain::Frequency, data.len(), num_data_packets);
    let total_payload = build_combined_payload(&header_bytes, &padded_data);
    let total_packets = HEADER_PACKETS + num_data_packets;

    for i in 0..total_packets {
        let offset = i * PAYLOAD_BYTES_PER_PACKET;
        let payload = [total_payload[offset], total_payload[offset + 1]];

        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, &mask_pool, &data_pool)?;

        let cipher = packet::generate_cipher(&config.seed, i as u64);
        freq_embed_cipher(&mut flat_mags, &indices.mask_indices, cipher, freq.delta);
        freq_embed_payload(&mut flat_mags, indices.data_index, payload, cipher, freq.delta);
    }

    // Apply dither to non-payload magnitude bins (raises spectral noise floor)
    if config.dither.enabled {
        let protected = dither::collect_protected_indices(
            &config.seed, total_packets, &mask_pool, &data_pool,
        )?;
        // Scale dither intensity proportionally: time_delta → freq.delta
        let freq_intensity = config.dither.intensity * (freq.delta / config.time_delta);
        dither::apply_freq_dither(&mut flat_mags, &protected, &config.seed, freq_intensity);
    }

    // Scatter modified magnitudes back
    for (flat_idx, &mag) in flat_mags.iter().enumerate() {
        let (frame, bin) = frame_bin_map[flat_idx];
        analysis.magnitudes[frame][bin] = mag;
    }

    let modified = freq_synthesize(&analysis, freq)?;
    samples[..analysis.used_len].copy_from_slice(&modified);

    Ok(EncodeMeta {
        num_packets: total_packets,
        original_len: data.len(),
        domain: EmbedDomain::Frequency,
    })
}

fn decode_full_freq(
    samples: &[f64],
    config: &EmbedConfig,
) -> Result<Vec<u8>, EmbedError> {
    let freq = &config.freq;
    let analysis = freq_analyze(samples, freq)?;
    let (flat_mags, mask_pool, data_pool, _frame_bin_map) =
        build_freq_pools(&analysis, &config.rms, freq, samples);

    // Extract header
    let header = extract_header_freq(&flat_mags, config, freq, &mask_pool, &data_pool)?;

    // Extract data packets
    let mut output = Vec::with_capacity(header.num_data_packets * PAYLOAD_BYTES_PER_PACKET);
    for i in 0..header.num_data_packets {
        let packet_num = (HEADER_PACKETS + i) as u64;
        let indices =
            packet::select_indices_from_pool(&config.seed, packet_num, &mask_pool, &data_pool)?;

        let cipher = freq_extract_cipher(&flat_mags, &indices.mask_indices, freq.delta);
        let payload = freq_extract_payload(&flat_mags, indices.data_index, cipher, freq.delta);
        output.extend_from_slice(&payload);
    }

    output.truncate(header.original_len);
    Ok(output)
}

// ── Header extraction helpers ────────────────────────────────────

/// Extract and validate the header from time-domain packets.
fn extract_header(
    samples: &[f64],
    config: &EmbedConfig,
    qim_config: &QimConfig,
    mask_pool: &[usize],
    data_pool: &[usize],
) -> Result<Header, EmbedError> {
    let mut header_bytes = Vec::with_capacity(HEADER_PACKETS * PAYLOAD_BYTES_PER_PACKET);

    for i in 0..HEADER_PACKETS {
        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, mask_pool, data_pool)?;

        let payload = packet::decode_packet_at(
            samples,
            &config.seed,
            i as u64,
            qim_config,
            &indices,
        )?;

        header_bytes.extend_from_slice(&payload);
    }

    let mut buf = [0u8; HEADER_SIZE];
    buf.copy_from_slice(&header_bytes[..HEADER_SIZE]);
    deserialize_header(&buf).map_err(|e| EmbedError::InvalidConfig(e.0))
}

/// Extract and validate the header from frequency-domain magnitude array.
fn extract_header_freq(
    flat_mags: &[f64],
    config: &EmbedConfig,
    freq: &FreqConfig,
    mask_pool: &[usize],
    data_pool: &[usize],
) -> Result<Header, EmbedError> {
    let mut header_bytes = Vec::with_capacity(HEADER_PACKETS * PAYLOAD_BYTES_PER_PACKET);

    for i in 0..HEADER_PACKETS {
        let indices =
            packet::select_indices_from_pool(&config.seed, i as u64, mask_pool, data_pool)?;

        let cipher = freq_extract_cipher(flat_mags, &indices.mask_indices, freq.delta);
        let payload = freq_extract_payload(flat_mags, indices.data_index, cipher, freq.delta);
        header_bytes.extend_from_slice(&payload);
    }

    let mut buf = [0u8; HEADER_SIZE];
    buf.copy_from_slice(&header_bytes[..HEADER_SIZE]);
    deserialize_header(&buf).map_err(|e| EmbedError::InvalidConfig(e.0))
}

/// Build a combined payload buffer: header + padded data, padded to even length.
fn build_combined_payload(header: &[u8; HEADER_SIZE], padded_data: &[u8]) -> Vec<u8> {
    let header_padded_len = HEADER_PACKETS * PAYLOAD_BYTES_PER_PACKET; // 14 bytes → 7 packets
    let mut combined = vec![0u8; header_padded_len + padded_data.len()];
    combined[..HEADER_SIZE].copy_from_slice(header);
    // Bytes HEADER_SIZE..header_padded_len are zero-padding for the header
    combined[header_padded_len..].copy_from_slice(padded_data);
    combined
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
            *byte = (i * 7 + 3) as u8; // Arbitrary but deterministic
        }
        seed
    }

    fn default_config() -> EmbedConfig {
        EmbedConfig {
            seed: test_seed(),
            ..Default::default()
        }
    }

    /// Generate a synthetic signal with both high and low power regions.
    /// The transition is aligned to the default segment size (4096) so that
    /// no segment straddles the boundary between loud and quiet.
    fn mixed_power_signal(num_segments: usize) -> Vec<f64> {
        let seg_size = 4096; // Must match RmsConfig::default().segment_size
        let loud_segments = num_segments / 2;
        let total_len = num_segments * seg_size;
        let transition = loud_segments * seg_size;

        (0..total_len)
            .map(|i| {
                let t = i as f64 / 44100.0;
                if i < transition {
                    0.9 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                } else {
                    0.001 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                }
            })
            .collect()
    }

    // ── RMS analysis tests ──────────────────────────────────────

    #[test]
    fn test_classify_rms() {
        let thresholds = RmsThresholds::default(); // high=0.1, low=0.01
        assert_eq!(classify_rms(0.5, &thresholds), PowerTier::High);
        assert_eq!(classify_rms(0.1, &thresholds), PowerTier::High);
        assert_eq!(classify_rms(0.05, &thresholds), PowerTier::Mid);
        assert_eq!(classify_rms(0.01, &thresholds), PowerTier::Mid);
        assert_eq!(classify_rms(0.005, &thresholds), PowerTier::Skip);
        assert_eq!(classify_rms(0.0, &thresholds), PowerTier::Skip);
    }

    #[test]
    fn test_analyze_segments_mixed() {
        let signal = mixed_power_signal(4); // 4 segments: 2 loud, 2 quiet
        let config = RmsConfig::default();
        let segments = analyze_segments(&signal, &config);

        assert_eq!(segments.len(), 4);

        // First segment (loud) should be High
        assert_eq!(segments[0].tier, PowerTier::High);

        // Last segment (quiet) should be Skip
        assert_eq!(segments[3].tier, PowerTier::Skip);
    }

    #[test]
    fn test_build_pools_excludes_skip() {
        let signal = mixed_power_signal(8); // 8 segments: 4 loud, 4 quiet
        let config = RmsConfig::default();
        let segments = analyze_segments(&signal, &config);
        let (mask_pool, data_pool) = build_eligible_pools(&segments, signal.len());

        // Transition is at exactly half = 4 * 4096 = 16384
        let transition = signal.len() / 2;
        for &idx in &mask_pool {
            assert!(idx < transition, "mask index {} should be in loud region", idx);
        }
        for &idx in &data_pool {
            assert!(
                idx + PAYLOAD_BITS_PER_PACKET <= transition,
                "data region at {} would extend into quiet region",
                idx
            );
        }

        assert!(mask_pool.len() >= MASK_SAMPLE_COUNT);
        assert!(!data_pool.is_empty());
    }

    // ── Time-domain embedding tests ─────────────────────────────

    #[test]
    fn test_time_domain_round_trip_synthetic() {
        let mut signal = vec![0.5f64; 100_000]; // Uniform high-power
        let config = default_config();
        let message = b"Hello, time domain!";

        let meta = encode_time(&mut signal, message, &config).unwrap();
        let decoded = decode_time(&signal, &meta, &config).unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_time_domain_rms_adaptive() {
        // 24 segments: 12 loud, 12 quiet. Transition at 12*4096 = 49152.
        let mut signal = mixed_power_signal(24);
        let original = signal.clone();
        let config = default_config();
        let message = b"RMS adaptive!";
        let transition = signal.len() / 2;

        let meta = encode_time(&mut signal, message, &config).unwrap();
        let decoded = decode_time(&signal, &meta, &config).unwrap();
        assert_eq!(&decoded, message);

        // Verify that quiet region (after transition) was NOT modified
        for i in transition..signal.len() {
            assert_eq!(
                signal[i], original[i],
                "sample {} in quiet region was modified",
                i
            );
        }
    }

    #[test]
    fn test_time_domain_various_payloads() {
        let config = default_config();

        for payload in [
            &b""[..],
            &b"A"[..],
            &b"AB"[..],
            &b"Short"[..],
            &b"A medium-length message for testing"[..],
        ] {
            // Use 500K samples to ensure negligible collision probability
            // for longer messages (35 bytes = 18 packets = ~450 samples used)
            let mut signal = vec![0.4f64; 500_000];
            let meta = encode_time(&mut signal, payload, &config).unwrap();
            let decoded = decode_time(&signal, &meta, &config).unwrap();
            assert_eq!(
                &decoded, payload,
                "round-trip failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    #[test]
    fn test_time_domain_wrong_seed() {
        let mut signal = vec![0.5f64; 100_000];
        let config = default_config();
        let message = b"Secret time data";

        let meta = encode_time(&mut signal, message, &config).unwrap();

        let mut wrong_config = config.clone();
        wrong_config.seed[0] ^= 0xFF;
        let decoded = decode_time(&signal, &meta, &wrong_config).unwrap();
        assert_ne!(&decoded, message);
    }

    // ── Frequency-domain embedding tests ────────────────────────

    #[test]
    fn test_freq_domain_round_trip_synthetic() {
        // Large constant signal — high RMS, all frames eligible
        let mut signal = vec![0.5f64; 16_384]; // 16 frames of 1024
        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.rms.thresholds.low = 0.0; // Accept everything
        let message = b"Freq!";

        let meta = encode_freq(&mut signal, message, &config).unwrap();
        let decoded = decode_freq(&signal, &meta, &config).unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_freq_domain_round_trip_sine() {
        // A real sine wave — tests that FFT analysis/synthesis preserves embedding
        let n = 1024 * 8;
        let mut signal: Vec<f64> = (0..n)
            .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.rms.thresholds.low = 0.0;
        let message = b"Hi";

        let meta = encode_freq(&mut signal, message, &config).unwrap();
        let decoded = decode_freq(&signal, &meta, &config).unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_freq_domain_wrong_seed() {
        let mut signal = vec![0.5f64; 16_384];
        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.rms.thresholds.low = 0.0;
        let message = b"Secret freq data";

        let meta = encode_freq(&mut signal, message, &config).unwrap();

        let mut wrong_config = config.clone();
        wrong_config.seed[0] ^= 0xFF;
        let decoded = decode_freq(&signal, &meta, &wrong_config).unwrap();
        assert_ne!(&decoded, message);
    }

    // ── Top-level dispatch tests ────────────────────────────────

    #[test]
    fn test_encode_decode_dispatch_time() {
        let mut signal = vec![0.5f64; 100_000];
        let config = default_config();
        let message = b"Dispatch time";

        let meta = encode(&mut signal, message, &config).unwrap();
        assert_eq!(meta.domain, EmbedDomain::Time);

        let decoded = decode(&signal, &meta, &config).unwrap();
        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_encode_decode_dispatch_freq() {
        let mut signal = vec![0.5f64; 16_384];
        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.rms.thresholds.low = 0.0;
        let message = b"Dispatch freq";

        let meta = encode(&mut signal, message, &config).unwrap();
        assert_eq!(meta.domain, EmbedDomain::Frequency);

        let decoded = decode(&signal, &meta, &config).unwrap();
        assert_eq!(&decoded, message);
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_empty_data() {
        let mut signal = vec![0.5f64; 10_000];
        let config = default_config();

        let meta = encode_time(&mut signal, &[], &config).unwrap();
        assert_eq!(meta.num_packets, 0);

        let decoded = decode_time(&signal, &meta, &config).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_all_skip_returns_error() {
        // Signal with RMS below the low threshold everywhere
        let mut signal = vec![0.0001f64; 10_000];
        let config = default_config();

        let result = encode_time(&mut signal, b"fail", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_freq_config() {
        let mut signal = vec![0.5f64; 4096];
        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.freq.window_size = 100; // Not power of 2

        let result = encode_freq(&mut signal, b"X", &config);
        assert!(result.is_err());
    }

    // ── Full pipeline (header-embedded) tests ─────────────────────

    #[test]
    fn test_header_serialize_deserialize() {
        let header = serialize_header(EmbedDomain::Time, 42, 21);
        let parsed = deserialize_header(&header).unwrap();
        assert_eq!(parsed.domain, EmbedDomain::Time);
        assert_eq!(parsed.original_len, 42);
        assert_eq!(parsed.num_data_packets, 21);

        let header_f = serialize_header(EmbedDomain::Frequency, 1000, 500);
        let parsed_f = deserialize_header(&header_f).unwrap();
        assert_eq!(parsed_f.domain, EmbedDomain::Frequency);
        assert_eq!(parsed_f.original_len, 1000);
        assert_eq!(parsed_f.num_data_packets, 500);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"NOPE");
        assert!(deserialize_header(&buf).is_err());
    }

    #[test]
    fn test_full_time_domain_round_trip() {
        let mut signal = vec![0.5f64; 500_000];
        let config = default_config();
        let message = b"Full pipeline time-domain test!";

        encode_full(&mut signal, message, &config).unwrap();
        let decoded = decode_full(&signal, &config).unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_full_time_domain_various_sizes() {
        let config = default_config();

        // Full pipeline adds 7 header packets, so total packet count is higher.
        // Use large signals (2M samples) to avoid packet collisions.
        for payload in [
            &b""[..],
            &b"X"[..],
            &b"AB"[..],
            &b"Hello StenoWav!"[..],
            &b"A somewhat longer message to verify multi-packet handling works"[..],
        ] {
            let mut signal = vec![0.4f64; 2_000_000];
            encode_full(&mut signal, payload, &config).unwrap();
            let decoded = decode_full(&signal, &config).unwrap();
            assert_eq!(
                &decoded, payload,
                "full round-trip failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    #[test]
    fn test_full_freq_domain_round_trip() {
        let mut signal = vec![0.5f64; 32_768]; // 32 frames of 1024
        let mut config = default_config();
        config.domain = EmbedDomain::Frequency;
        config.rms.thresholds.low = 0.0;
        let message = b"Freq full pipeline!";

        encode_full(&mut signal, message, &config).unwrap();
        let decoded = decode_full(&signal, &config).unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_full_wrong_seed_fails_header() {
        let mut signal = vec![0.5f64; 500_000];
        let config = default_config();
        let message = b"Secrets";

        encode_full(&mut signal, message, &config).unwrap();

        // Wrong seed should fail header validation
        let mut wrong_config = config.clone();
        wrong_config.seed[0] ^= 0xFF;
        let result = decode_full(&signal, &wrong_config);
        assert!(result.is_err(), "wrong seed should fail header check");
    }

    #[test]
    fn test_full_time_domain_parameter_sweep() {
        // Test with different deltas
        for &delta in &[0.003, 0.005, 0.01, 0.02] {
            let mut config = default_config();
            config.time_delta = delta;

            let mut signal = vec![0.5f64; 500_000];
            let message = b"Delta sweep";
            encode_full(&mut signal, message, &config).unwrap();
            let decoded = decode_full(&signal, &config).unwrap();
            assert_eq!(
                &decoded, message,
                "failed at delta={}",
                delta
            );
        }
    }
}
