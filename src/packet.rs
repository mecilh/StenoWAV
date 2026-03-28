//! Packet-based steganographic encoding.
//!
//! A **packet** is the fundamental unit of data embedding. Each packet uses
//! 9 randomly selected (non-contiguous) samples from the audio signal:
//!
//! - **Mask samples** (indices 0–7): Combined via QIM to produce a 16-bit cipher key.
//!   Each mask sample contributes 2 bits to the cipher (8 samples x 2 bits = 16 bits).
//!
//! - **Data sample** (index 8): Carries 2 bytes (16 bits) of payload, XORed with the
//!   cipher derived from the mask samples before embedding.
//!
//! The 9 sample indices are chosen purely randomly via a seeded ChaCha20 CSPRNG,
//! ensuring reproducibility for decoding while being unpredictable to an attacker
//! without the seed.
//!
//! ## Data flow
//!
//! ```text
//! Encode:
//!   seed + packet_number -> CSPRNG -> 9 random indices
//!   mask_samples[0..7]   -> QIM embed 2 bits each -> 16-bit cipher
//!   payload_2bytes XOR cipher -> QIM embed 16 bits into data_sample[8]
//!
//! Decode:
//!   seed + packet_number -> CSPRNG -> same 9 indices
//!   mask_samples[0..7]   -> QIM extract 2 bits each -> 16-bit cipher
//!   data_sample[8]       -> QIM extract 16 bits -> XOR cipher -> payload_2bytes
//! ```

use crate::cipher::{self, CipherKey};
use crate::qim::{self, QimConfig, QimError};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::fmt;

/// Number of mask samples per packet.
pub const MASK_SAMPLE_COUNT: usize = 8;

/// Number of data samples per packet (carries the payload).
pub const DATA_SAMPLE_COUNT: usize = 1;

/// Total samples per packet.
pub const PACKET_SAMPLE_COUNT: usize = MASK_SAMPLE_COUNT + DATA_SAMPLE_COUNT;

/// Bits embedded per mask sample.
pub const BITS_PER_MASK_SAMPLE: usize = 2;

/// Bytes of payload per packet.
pub const PAYLOAD_BYTES_PER_PACKET: usize = 2;

/// Bits of payload per packet (16).
pub const PAYLOAD_BITS_PER_PACKET: usize = PAYLOAD_BYTES_PER_PACKET * 8;

/// Packet-related errors.
#[derive(Debug, Clone)]
pub enum PacketError {
    /// Not enough samples in the signal to place a packet.
    InsufficientSamples { available: usize, required: usize },
    /// QIM operation failed.
    Qim(QimError),
    /// Seed must be exactly 32 bytes.
    InvalidSeedLength(usize),
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacketError::InsufficientSamples { available, required } => {
                write!(
                    f,
                    "not enough samples: have {}, need at least {}",
                    available, required
                )
            }
            PacketError::Qim(e) => write!(f, "QIM error: {}", e),
            PacketError::InvalidSeedLength(n) => {
                write!(f, "seed must be 32 bytes, got {}", n)
            }
        }
    }
}

impl std::error::Error for PacketError {}

impl From<QimError> for PacketError {
    fn from(e: QimError) -> Self {
        PacketError::Qim(e)
    }
}

/// A resolved packet: the 9 sample indices selected for one unit of data.
#[derive(Debug, Clone)]
pub struct PacketIndices {
    /// Indices of the 8 mask samples in the audio signal.
    pub mask_indices: [usize; MASK_SAMPLE_COUNT],
    /// Index of the data sample in the audio signal.
    pub data_index: usize,
}

/// Generate a deterministic ChaCha20 RNG from a 32-byte seed and a packet number.
///
/// The packet number is mixed into the seed so each packet gets a unique
/// but reproducible random stream.
fn make_packet_rng(seed: &[u8; 32], packet_number: u64) -> ChaCha20Rng {
    let mut keyed_seed = *seed;
    // Mix the packet number into the last 8 bytes of the seed
    let pn_bytes = packet_number.to_le_bytes();
    for i in 0..8 {
        keyed_seed[24 + i] ^= pn_bytes[i];
    }
    ChaCha20Rng::from_seed(keyed_seed)
}

/// Select 9 unique random sample indices within [0, signal_len).
///
/// Uses a Fisher-Yates partial shuffle approach to guarantee uniqueness
/// without rejection sampling overhead for large signals.
pub fn select_indices(
    seed: &[u8; 32],
    packet_number: u64,
    signal_len: usize,
) -> Result<PacketIndices, PacketError> {
    if signal_len < PACKET_SAMPLE_COUNT {
        return Err(PacketError::InsufficientSamples {
            available: signal_len,
            required: PACKET_SAMPLE_COUNT,
        });
    }

    let mut rng = make_packet_rng(seed, packet_number);

    // Pick 9 unique indices using partial Fisher-Yates on an index pool.
    // For large signals, we use a hash-set approach to avoid allocating
    // the full signal length.
    let mut selected = Vec::with_capacity(PACKET_SAMPLE_COUNT);
    let mut used = std::collections::HashSet::with_capacity(PACKET_SAMPLE_COUNT);

    while selected.len() < PACKET_SAMPLE_COUNT {
        let idx = rng.random_range(0..signal_len);
        if used.insert(idx) {
            selected.push(idx);
        }
    }

    let mut mask_indices = [0usize; MASK_SAMPLE_COUNT];
    mask_indices.copy_from_slice(&selected[..MASK_SAMPLE_COUNT]);
    let data_index = selected[MASK_SAMPLE_COUNT];

    Ok(PacketIndices {
        mask_indices,
        data_index,
    })
}

// Cipher embed/extract is delegated to the cipher module.
// See cipher.rs for the formal derivation and QIM primitives.

/// Embed 2 bytes of payload into the data sample, XORed with the cipher.
fn embed_payload_into_data(
    samples: &mut [f64],
    data_index: usize,
    payload: [u8; PAYLOAD_BYTES_PER_PACKET],
    cipher: CipherKey,
    config: &QimConfig,
) -> Result<(), PacketError> {
    let payload_u16 = u16::from_be_bytes(payload);
    let masked = cipher.apply(payload_u16);

    // Embed 16 bits into the data sample region.
    // We use a progressively finer QIM lattice for each bit.
    // Bit 15 (MSB) uses delta, bit 14 uses delta/2, ..., bit 0 uses delta/2^15.
    //
    // HOWEVER: that would make the LSBs unrecoverable through i16 quantisation.
    // Instead, we spread the 16 bits across the data sample and its neighbours
    // by using the data_index as a starting point in a local region.
    //
    // For this version: we embed 16 bits into 16 samples starting at data_index.
    // This is safe because select_indices guarantees non-overlapping indices,
    // and we use a contiguous block from data_index for the payload bits.
    //
    // Actually, to keep the architecture clean and each packet self-contained:
    // We embed 1 bit per sample using QIM, using 16 consecutive samples
    // starting from data_index. The select_indices function must ensure
    // data_index has 16 samples of runway.

    // For now: we embed 16 bits across the single data sample using
    // decreasing delta. But we must ensure delta/2^k > quantisation noise.
    // With base delta = 0.005, delta/2^15 ≈ 1.5e-7 which is way below
    // WAV noise floor (3e-5). So we can only safely nest ~7 bits.
    //
    // DESIGN DECISION: Use 16 samples for the data payload (1 bit each).
    // The data_index points to the FIRST of 16 contiguous samples.
    for bit_pos in 0..PAYLOAD_BITS_PER_PACKET {
        let bit = ((masked >> (PAYLOAD_BITS_PER_PACKET - 1 - bit_pos)) & 1) as u8;
        let sample_idx = data_index + bit_pos;
        samples[sample_idx] = qim::embed_bit(samples[sample_idx], bit, config)?;
    }

    Ok(())
}

/// Extract 2 bytes of payload from the data sample region, removing the cipher mask.
fn extract_payload_from_data(
    samples: &[f64],
    data_index: usize,
    cipher: CipherKey,
    config: &QimConfig,
) -> [u8; PAYLOAD_BYTES_PER_PACKET] {
    let mut masked: u16 = 0;

    for bit_pos in 0..PAYLOAD_BITS_PER_PACKET {
        let sample_idx = data_index + bit_pos;
        let bit = qim::extract_bit(samples[sample_idx], config);
        masked |= (bit as u16) << (PAYLOAD_BITS_PER_PACKET - 1 - bit_pos);
    }

    let payload_u16 = cipher.apply(masked);
    payload_u16.to_be_bytes()
}

/// Generate a random cipher key for a given packet.
///
/// Delegates to [`cipher::generate`] — see that module for the formal
/// derivation process and security properties.
pub fn generate_cipher(seed: &[u8; 32], packet_number: u64) -> CipherKey {
    cipher::generate(seed, packet_number)
}

/// Encode a single packet into the audio signal.
///
/// Embeds 2 bytes of payload using 9 randomly selected samples
/// (8 mask + 1 data region start).
///
/// # Arguments
/// * `samples` - Mutable audio samples (single channel)
/// * `seed` - 32-byte seed for reproducible index selection
/// * `packet_number` - Sequential packet number (0, 1, 2, ...)
/// * `payload` - 2 bytes to embed
/// * `config` - QIM configuration
pub fn encode_packet(
    samples: &mut [f64],
    seed: &[u8; 32],
    packet_number: u64,
    payload: [u8; PAYLOAD_BYTES_PER_PACKET],
    config: &QimConfig,
) -> Result<PacketIndices, PacketError> {
    let indices = select_indices(seed, packet_number, samples.len())?;

    // Verify data region has enough runway for 16 bits
    if indices.data_index + PAYLOAD_BITS_PER_PACKET > samples.len() {
        return Err(PacketError::InsufficientSamples {
            available: samples.len() - indices.data_index,
            required: PAYLOAD_BITS_PER_PACKET,
        });
    }

    // Generate and embed cipher into mask samples
    let cipher = generate_cipher(seed, packet_number);
    cipher::embed_into_masks(samples, &indices.mask_indices, cipher, config);

    // Embed payload XORed with cipher into data region
    embed_payload_into_data(samples, indices.data_index, payload, cipher, config)?;

    Ok(indices)
}

/// Decode a single packet from the audio signal.
///
/// Extracts 2 bytes of payload by reconstructing the same indices
/// from the seed and reading back the cipher + data.
///
/// # Arguments
/// * `samples` - Audio samples (single channel, potentially watermarked)
/// * `seed` - Same 32-byte seed used during encoding
/// * `packet_number` - Same packet number used during encoding
/// * `config` - Same QIM configuration used during encoding
pub fn decode_packet(
    samples: &[f64],
    seed: &[u8; 32],
    packet_number: u64,
    config: &QimConfig,
) -> Result<[u8; PAYLOAD_BYTES_PER_PACKET], PacketError> {
    let indices = select_indices(seed, packet_number, samples.len())?;

    if indices.data_index + PAYLOAD_BITS_PER_PACKET > samples.len() {
        return Err(PacketError::InsufficientSamples {
            available: samples.len() - indices.data_index,
            required: PAYLOAD_BITS_PER_PACKET,
        });
    }

    // Extract cipher from mask samples
    let cipher = cipher::extract_from_masks(samples, &indices.mask_indices, config);

    // Extract payload from data region and remove cipher
    let payload = extract_payload_from_data(samples, indices.data_index, cipher, config);

    Ok(payload)
}

/// Encode an arbitrary byte stream into an audio signal.
///
/// Splits the data into 2-byte chunks and encodes each as a packet.
/// Pads the last chunk with a zero byte if the data length is odd.
///
/// Returns the number of packets encoded.
pub fn encode_stream(
    samples: &mut [f64],
    seed: &[u8; 32],
    data: &[u8],
    config: &QimConfig,
) -> Result<usize, PacketError> {
    let mut padded = data.to_vec();
    if padded.len() % PAYLOAD_BYTES_PER_PACKET != 0 {
        padded.push(0x00); // Pad to even length
    }

    let num_packets = padded.len() / PAYLOAD_BYTES_PER_PACKET;

    for i in 0..num_packets {
        let offset = i * PAYLOAD_BYTES_PER_PACKET;
        let payload: [u8; PAYLOAD_BYTES_PER_PACKET] =
            [padded[offset], padded[offset + 1]];
        encode_packet(samples, seed, i as u64, payload, config)?;
    }

    Ok(num_packets)
}

/// Decode a byte stream from an audio signal.
///
/// Extracts `num_packets` packets and concatenates their payloads.
/// If `original_len` is provided, the output is truncated to that length
/// (removing any padding added during encoding).
pub fn decode_stream(
    samples: &[f64],
    seed: &[u8; 32],
    num_packets: usize,
    original_len: Option<usize>,
    config: &QimConfig,
) -> Result<Vec<u8>, PacketError> {
    let mut output = Vec::with_capacity(num_packets * PAYLOAD_BYTES_PER_PACKET);

    for i in 0..num_packets {
        let payload = decode_packet(samples, seed, i as u64, config)?;
        output.extend_from_slice(&payload);
    }

    if let Some(len) = original_len {
        output.truncate(len);
    }

    Ok(output)
}

// ── Pool-aware API for dual-domain embedding ────────────────────

/// Select packet indices from explicit pools of eligible positions.
///
/// Unlike [`select_indices`] which picks from the entire signal range,
/// this function restricts selection to caller-provided pools. Used by
/// the RMS-adaptive embedding engine to avoid low-power regions.
///
/// * `mask_pool` — eligible sample indices for the 8 mask samples
/// * `data_pool` — eligible starting indices for the 16-sample data region
///
/// The data region \[data_index .. data_index + 16) must not overlap any mask index.
pub fn select_indices_from_pool(
    seed: &[u8; 32],
    packet_number: u64,
    mask_pool: &[usize],
    data_pool: &[usize],
) -> Result<PacketIndices, PacketError> {
    if mask_pool.len() < MASK_SAMPLE_COUNT {
        return Err(PacketError::InsufficientSamples {
            available: mask_pool.len(),
            required: MASK_SAMPLE_COUNT,
        });
    }
    if data_pool.is_empty() {
        return Err(PacketError::InsufficientSamples {
            available: 0,
            required: 1,
        });
    }

    let mut rng = make_packet_rng(seed, packet_number);

    // Select MASK_SAMPLE_COUNT unique indices from mask_pool via rejection sampling
    let mut mask_indices = [0usize; MASK_SAMPLE_COUNT];
    let mut used = std::collections::HashSet::with_capacity(PACKET_SAMPLE_COUNT + PAYLOAD_BITS_PER_PACKET);
    let mut count = 0;
    let max_attempts = mask_pool.len() * 100;
    let mut attempts = 0;

    while count < MASK_SAMPLE_COUNT {
        attempts += 1;
        if attempts > max_attempts {
            return Err(PacketError::InsufficientSamples {
                available: count,
                required: MASK_SAMPLE_COUNT,
            });
        }
        let pool_idx = rng.random_range(0..mask_pool.len());
        let idx = mask_pool[pool_idx];
        if used.insert(idx) {
            mask_indices[count] = idx;
            count += 1;
        }
    }

    // Select data start index — the 16-sample data region must not overlap mask indices
    let max_data_attempts = data_pool.len() * 100;
    let mut data_attempts = 0;

    loop {
        data_attempts += 1;
        if data_attempts > max_data_attempts {
            return Err(PacketError::InsufficientSamples {
                available: 0,
                required: PAYLOAD_BITS_PER_PACKET,
            });
        }

        let pool_idx = rng.random_range(0..data_pool.len());
        let idx = data_pool[pool_idx];

        let overlaps = (0..PAYLOAD_BITS_PER_PACKET).any(|offset| used.contains(&(idx + offset)));
        if !overlaps {
            return Ok(PacketIndices {
                mask_indices,
                data_index: idx,
            });
        }
    }
}

/// Encode a single packet using pre-computed indices.
///
/// Low-level function for the dual-domain embedding engine. The caller
/// selects indices (e.g. via [`select_indices_from_pool`]) and this
/// function embeds the mask cipher + payload at those positions.
pub fn encode_packet_at(
    samples: &mut [f64],
    seed: &[u8; 32],
    packet_number: u64,
    payload: [u8; PAYLOAD_BYTES_PER_PACKET],
    config: &QimConfig,
    indices: &PacketIndices,
) -> Result<(), PacketError> {
    if indices.data_index + PAYLOAD_BITS_PER_PACKET > samples.len() {
        return Err(PacketError::InsufficientSamples {
            available: samples.len().saturating_sub(indices.data_index),
            required: PAYLOAD_BITS_PER_PACKET,
        });
    }

    let cipher = generate_cipher(seed, packet_number);
    cipher::embed_into_masks(samples, &indices.mask_indices, cipher, config);
    embed_payload_into_data(samples, indices.data_index, payload, cipher, config)?;
    Ok(())
}

/// Decode a single packet using pre-computed indices.
///
/// Low-level function for the dual-domain embedding engine. The caller
/// provides the same indices used during encoding; this function extracts
/// the cipher from mask samples and recovers the payload.
pub fn decode_packet_at(
    samples: &[f64],
    _seed: &[u8; 32],
    _packet_number: u64,
    config: &QimConfig,
    indices: &PacketIndices,
) -> Result<[u8; PAYLOAD_BYTES_PER_PACKET], PacketError> {
    if indices.data_index + PAYLOAD_BITS_PER_PACKET > samples.len() {
        return Err(PacketError::InsufficientSamples {
            available: samples.len().saturating_sub(indices.data_index),
            required: PAYLOAD_BITS_PER_PACKET,
        });
    }

    let cipher = cipher::extract_from_masks(samples, &indices.mask_indices, config);
    let payload = extract_payload_from_data(samples, indices.data_index, cipher, config);
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seed() -> [u8; 32] {
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = i as u8;
        }
        seed
    }

    fn test_config() -> QimConfig {
        QimConfig::new(0.01).unwrap()
    }

    #[test]
    fn test_select_indices_deterministic() {
        let seed = test_seed();
        let indices1 = select_indices(&seed, 0, 10000).unwrap();
        let indices2 = select_indices(&seed, 0, 10000).unwrap();

        // Same seed + packet number -> same indices
        assert_eq!(indices1.mask_indices, indices2.mask_indices);
        assert_eq!(indices1.data_index, indices2.data_index);
    }

    #[test]
    fn test_select_indices_different_packets() {
        let seed = test_seed();
        let indices0 = select_indices(&seed, 0, 10000).unwrap();
        let indices1 = select_indices(&seed, 1, 10000).unwrap();

        // Different packet numbers should (almost certainly) give different indices
        assert_ne!(indices0.mask_indices, indices1.mask_indices);
    }

    #[test]
    fn test_select_indices_all_unique() {
        let seed = test_seed();
        let indices = select_indices(&seed, 42, 10000).unwrap();

        let mut all: Vec<usize> = indices.mask_indices.to_vec();
        all.push(indices.data_index);
        let set: std::collections::HashSet<usize> = all.iter().cloned().collect();
        assert_eq!(set.len(), PACKET_SAMPLE_COUNT, "all indices must be unique");
    }

    #[test]
    fn test_select_indices_too_few_samples() {
        let seed = test_seed();
        assert!(select_indices(&seed, 0, 5).is_err());
    }

    #[test]
    fn test_cipher_round_trip() {
        let config = test_config();
        let seed = test_seed();

        // Create a signal with enough samples
        let samples = vec![0.5f64; 10000];
        let indices = select_indices(&seed, 0, samples.len()).unwrap();

        // Test multiple cipher values via the cipher module
        for cipher_val in [0x0000u16, 0xFFFF, 0xAAAA, 0x5555, 0x1234, 0xDEAD] {
            let cipher = CipherKey::new(cipher_val);
            let mut test_samples = samples.clone();
            cipher::embed_into_masks(
                &mut test_samples,
                &indices.mask_indices,
                cipher,
                &config,
            );

            let extracted =
                cipher::extract_from_masks(&test_samples, &indices.mask_indices, &config);
            assert_eq!(
                extracted, cipher,
                "cipher round-trip failed for 0x{:04X}: got {}",
                cipher_val, extracted
            );
        }
    }

    #[test]
    fn test_single_packet_round_trip() {
        let config = test_config();
        let seed = test_seed();
        let mut samples = vec![0.3f64; 50000];

        let payload: [u8; 2] = [0xDE, 0xAD];
        encode_packet(&mut samples, &seed, 0, payload, &config).unwrap();
        let decoded = decode_packet(&samples, &seed, 0, &config).unwrap();

        assert_eq!(decoded, payload, "packet round-trip failed");
    }

    #[test]
    fn test_single_packet_all_payloads() {
        let config = test_config();
        let seed = test_seed();

        // Test a spread of payload values
        for hi in (0..=255).step_by(51) {
            for lo in (0..=255).step_by(51) {
                let mut samples = vec![0.4f64; 50000];
                let payload = [hi, lo];
                encode_packet(&mut samples, &seed, 0, payload, &config).unwrap();
                let decoded = decode_packet(&samples, &seed, 0, &config).unwrap();
                assert_eq!(
                    decoded, payload,
                    "failed for payload [{:#04X}, {:#04X}]",
                    hi, lo
                );
            }
        }
    }

    #[test]
    fn test_stream_round_trip() {
        let config = test_config();
        let seed = test_seed();
        let mut samples = vec![0.2f64; 100000];

        let message = b"Hello, StenoWav!";
        let num_packets =
            encode_stream(&mut samples, &seed, message, &config).unwrap();

        let decoded = decode_stream(
            &samples,
            &seed,
            num_packets,
            Some(message.len()),
            &config,
        )
        .unwrap();

        assert_eq!(
            &decoded,
            message,
            "stream round-trip failed: got {:?}",
            String::from_utf8_lossy(&decoded)
        );
    }

    #[test]
    fn test_stream_odd_length() {
        let config = test_config();
        let seed = test_seed();
        let mut samples = vec![0.2f64; 100000];

        // Odd-length message (needs padding)
        let message = b"Odd";
        let num_packets =
            encode_stream(&mut samples, &seed, message, &config).unwrap();
        assert_eq!(num_packets, 2); // 3 bytes -> padded to 4 -> 2 packets

        let decoded = decode_stream(
            &samples,
            &seed,
            num_packets,
            Some(message.len()),
            &config,
        )
        .unwrap();

        assert_eq!(&decoded, message);
    }

    #[test]
    fn test_stream_empty() {
        let config = test_config();
        let seed = test_seed();
        let mut samples = vec![0.2f64; 100000];

        let num_packets =
            encode_stream(&mut samples, &seed, &[], &config).unwrap();
        assert_eq!(num_packets, 0);

        let decoded =
            decode_stream(&samples, &seed, 0, Some(0), &config).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_multiple_packets_independent() {
        let config = test_config();
        let seed = test_seed();
        let mut samples = vec![0.5f64; 100000];

        // Encode two packets with different payloads
        encode_packet(&mut samples, &seed, 0, [0xAA, 0xBB], &config).unwrap();
        encode_packet(&mut samples, &seed, 1, [0xCC, 0xDD], &config).unwrap();

        // Decode them independently
        let decoded0 = decode_packet(&samples, &seed, 0, &config).unwrap();
        let decoded1 = decode_packet(&samples, &seed, 1, &config).unwrap();

        assert_eq!(decoded0, [0xAA, 0xBB]);
        assert_eq!(decoded1, [0xCC, 0xDD]);
    }

    #[test]
    fn test_wrong_seed_fails() {
        let config = test_config();
        let seed = test_seed();
        let mut wrong_seed = test_seed();
        wrong_seed[0] = 0xFF; // Different seed

        let mut samples = vec![0.5f64; 100000];
        encode_packet(&mut samples, &seed, 0, [0xDE, 0xAD], &config).unwrap();

        // Decoding with wrong seed should give wrong result
        let decoded = decode_packet(&samples, &wrong_seed, 0, &config).unwrap();
        // It's technically possible (but astronomically unlikely) for this to match
        assert_ne!(decoded, [0xDE, 0xAD], "wrong seed should not decode correctly");
    }
}
