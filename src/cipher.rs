//! Cipher derivation from mask samples.
//!
//! This module formalises the process by which 8 mask samples in a packet
//! produce a 16-bit cipher key used to encrypt the payload before embedding.
//!
//! ## Cipher derivation
//!
//! Each packet selects 8 mask samples at CSPRNG-determined positions.
//! Each mask sample encodes 2 bits via **mod-4 QIM** (quantisation index
//! modulation with M=4), giving 8 × 2 = 16 bits total — the cipher key.
//!
//! ```text
//!  mask_sample[0]  ──► QIM mod 4 ──► 2 bits (bits 14–15 of cipher)
//!  mask_sample[1]  ──► QIM mod 4 ──► 2 bits (bits 12–13)
//!  mask_sample[2]  ──► QIM mod 4 ──► 2 bits (bits 10–11)
//!  mask_sample[3]  ──► QIM mod 4 ──► 2 bits (bits  8–9)
//!  mask_sample[4]  ──► QIM mod 4 ──► 2 bits (bits  6–7)
//!  mask_sample[5]  ──► QIM mod 4 ──► 2 bits (bits  4–5)
//!  mask_sample[6]  ──► QIM mod 4 ──► 2 bits (bits  2–3)
//!  mask_sample[7]  ──► QIM mod 4 ──► 2 bits (bits  0–1)
//! ```
//!
//! The cipher is then XORed with the 16-bit payload before data embedding,
//! making extraction impossible without knowing both the seed (to find
//! mask positions) and the QIM delta (to read the mask values).
//!
//! ## Cipher generation vs extraction
//!
//! - **Encoding**: A random 16-bit cipher is generated from a separate
//!   ChaCha20 stream (seed with first byte flipped + packet number).
//!   This cipher is embedded into the 8 mask samples via mod-4 QIM.
//!
//! - **Decoding**: The same mask sample positions are recomputed from
//!   the seed, and the cipher is extracted via mod-4 QIM extraction.
//!   No separate cipher stream is needed — the cipher lives in the signal.
//!
//! ## Security properties
//!
//! - Without the seed, an attacker cannot locate the mask samples.
//! - Without the QIM delta, an attacker cannot extract the mod-4 residues.
//! - The cipher is unique per packet (different CSPRNG stream per packet number).
//! - The XOR of payload with cipher means identical payloads in different
//!   packets produce different embedded bit patterns.

use crate::qim::QimConfig;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Number of mask samples that contribute to the cipher.
pub const CIPHER_MASK_COUNT: usize = 8;

/// Bits contributed by each mask sample (mod-4 QIM → 2 bits).
pub const BITS_PER_MASK: usize = 2;

/// Total cipher width in bits (8 × 2 = 16).
pub const CIPHER_BITS: usize = CIPHER_MASK_COUNT * BITS_PER_MASK;

/// Modulus for multi-bit QIM on mask samples.
pub const MASK_QIM_MODULUS: u8 = 4;

/// A 16-bit per-packet cipher key derived from mask samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CipherKey(pub u16);

impl CipherKey {
    /// Create a cipher key from a raw u16 value.
    pub fn new(value: u16) -> Self {
        CipherKey(value)
    }

    /// Apply the cipher to a 16-bit payload (XOR encryption/decryption).
    ///
    /// XOR is its own inverse: `apply(apply(payload)) == payload`.
    pub fn apply(&self, payload: u16) -> u16 {
        payload ^ self.0
    }

    /// Get the raw cipher value.
    pub fn value(&self) -> u16 {
        self.0
    }

    /// Extract the 2-bit symbol for a given mask index (0–7).
    ///
    /// Returns the bits that mask sample `i` encodes, in range [0, 3].
    pub fn symbol_for_mask(&self, mask_index: usize) -> u8 {
        assert!(mask_index < CIPHER_MASK_COUNT, "mask index out of range");
        let bit_offset = (CIPHER_MASK_COUNT - 1 - mask_index) * BITS_PER_MASK;
        ((self.0 >> bit_offset) & 0x03) as u8
    }

    /// Reconstruct a cipher key from 8 individual 2-bit symbols.
    ///
    /// `symbols[i]` is the mod-4 residue extracted from mask sample `i`.
    pub fn from_symbols(symbols: &[u8; CIPHER_MASK_COUNT]) -> Self {
        let mut value: u16 = 0;
        for i in 0..CIPHER_MASK_COUNT {
            debug_assert!(symbols[i] < MASK_QIM_MODULUS, "symbol out of range");
            let bit_offset = (CIPHER_MASK_COUNT - 1 - i) * BITS_PER_MASK;
            value |= (symbols[i] as u16) << bit_offset;
        }
        CipherKey(value)
    }
}

impl std::fmt::Display for CipherKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CipherKey(0x{:04X})", self.0)
    }
}

// ── Cipher generation ────────────────────────────────────────────

/// Generate a deterministic cipher key for a given packet.
///
/// Uses a ChaCha20 CSPRNG seeded from the master key, with the first byte
/// flipped to produce a stream independent from the index-selection RNG.
/// The packet number is mixed into the seed for per-packet uniqueness.
pub fn generate(seed: &[u8; 32], packet_number: u64) -> CipherKey {
    let mut cipher_seed = *seed;
    // Flip the first byte to create a separate stream from index selection
    cipher_seed[0] ^= 0xFF;

    // Mix packet number into the last 8 bytes
    let pn_bytes = packet_number.to_le_bytes();
    for i in 0..8 {
        cipher_seed[24 + i] ^= pn_bytes[i];
    }

    let mut rng = ChaCha20Rng::from_seed(cipher_seed);
    CipherKey(rng.random_range(0..=u16::MAX))
}

// ── Embed / extract cipher into/from mask samples ────────────────

/// Embed a cipher key into 8 mask samples using mod-4 QIM.
///
/// Each mask sample is quantised so that its quantisation index has the
/// correct residue mod 4, encoding 2 bits of the cipher.
pub fn embed_into_masks(
    samples: &mut [f64],
    mask_indices: &[usize; CIPHER_MASK_COUNT],
    cipher: CipherKey,
    config: &QimConfig,
) {
    for i in 0..CIPHER_MASK_COUNT {
        let symbol = cipher.symbol_for_mask(i);
        let idx = mask_indices[i];
        samples[idx] = embed_symbol(samples[idx], symbol, MASK_QIM_MODULUS, config.delta);
    }
}

/// Extract a cipher key from 8 mask samples using mod-4 QIM.
///
/// Reads the mod-4 residue of the quantisation index at each mask position
/// and reconstructs the 16-bit cipher key.
pub fn extract_from_masks(
    samples: &[f64],
    mask_indices: &[usize; CIPHER_MASK_COUNT],
    config: &QimConfig,
) -> CipherKey {
    let mut symbols = [0u8; CIPHER_MASK_COUNT];
    for i in 0..CIPHER_MASK_COUNT {
        let idx = mask_indices[i];
        symbols[i] = extract_symbol(samples[idx], MASK_QIM_MODULUS, config.delta);
    }
    CipherKey::from_symbols(&symbols)
}

// ── Non-negative variants for frequency-domain magnitudes ────────

/// Embed a cipher key into magnitude bins using non-negative mod-4 QIM.
///
/// Frequency magnitudes are always ≥ 0, so quantisation indices are
/// constrained to ≥ 0 to prevent sign-loss through FFT round-trips.
pub fn embed_into_magnitudes(
    flat_mags: &mut [f64],
    mask_indices: &[usize; CIPHER_MASK_COUNT],
    cipher: CipherKey,
    delta: f64,
) {
    for i in 0..CIPHER_MASK_COUNT {
        let symbol = cipher.symbol_for_mask(i);
        let idx = mask_indices[i];
        flat_mags[idx] = embed_symbol_nonneg(flat_mags[idx], symbol, MASK_QIM_MODULUS, delta);
    }
}

/// Extract a cipher key from magnitude bins using non-negative mod-4 QIM.
pub fn extract_from_magnitudes(
    flat_mags: &[f64],
    mask_indices: &[usize; CIPHER_MASK_COUNT],
    delta: f64,
) -> CipherKey {
    let mut symbols = [0u8; CIPHER_MASK_COUNT];
    for i in 0..CIPHER_MASK_COUNT {
        let idx = mask_indices[i];
        symbols[i] = extract_symbol_nonneg(flat_mags[idx], MASK_QIM_MODULUS, delta);
    }
    CipherKey::from_symbols(&symbols)
}

// ── Multi-bit QIM primitives ─────────────────────────────────────
//
// These are the core quantisation operations. They embed/extract a symbol
// in [0, m) by quantising to the nearest lattice point with the correct
// residue mod m.

/// Embed a symbol in [0, m) into a sample via mod-m QIM.
pub(crate) fn embed_symbol(sample: f64, symbol: u8, m: u8, delta: f64) -> f64 {
    let q = (sample / delta).round() as i64;
    let current_mod = q.rem_euclid(m as i64) as u8;

    if current_mod == symbol {
        return q as f64 * delta;
    }

    let diff = (symbol as i64 - current_mod as i64).rem_euclid(m as i64);
    let candidate_up = q + diff;
    let candidate_down = q + diff - m as i64;

    let dist_up = (candidate_up as f64 * delta - sample).abs();
    let dist_down = (candidate_down as f64 * delta - sample).abs();

    if dist_up <= dist_down {
        candidate_up as f64 * delta
    } else {
        candidate_down as f64 * delta
    }
}

/// Extract a symbol in [0, m) from a sample via mod-m QIM.
pub(crate) fn extract_symbol(sample: f64, m: u8, delta: f64) -> u8 {
    let q = (sample / delta).round() as i64;
    q.rem_euclid(m as i64) as u8
}

/// Embed a symbol using non-negative QIM (for frequency magnitudes).
pub(crate) fn embed_symbol_nonneg(mag: f64, symbol: u8, m: u8, delta: f64) -> f64 {
    let q = (mag / delta).round().max(0.0) as i64;
    let current_mod = (q as u64 % m as u64) as u8;

    if current_mod == symbol {
        return q as f64 * delta;
    }

    let diff = ((symbol as i64 - current_mod as i64).rem_euclid(m as i64)) as i64;
    let candidate_up = q + diff;
    let candidate_down = q + diff - m as i64;

    let up_val = candidate_up as f64 * delta;

    if candidate_down >= 0 {
        let down_val = candidate_down as f64 * delta;
        if (up_val - mag).abs() <= (down_val - mag).abs() {
            up_val
        } else {
            down_val
        }
    } else {
        up_val
    }
}

/// Extract a symbol using non-negative QIM (for frequency magnitudes).
pub(crate) fn extract_symbol_nonneg(mag: f64, m: u8, delta: f64) -> u8 {
    let q = (mag / delta).round().max(0.0) as u64;
    (q % m as u64) as u8
}

// ── Tests ────────────────────────────────────────────────────────

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

    #[test]
    fn test_cipher_key_apply_is_involution() {
        let key = CipherKey::new(0xABCD);
        let payload: u16 = 0x1234;
        let encrypted = key.apply(payload);
        let decrypted = key.apply(encrypted);
        assert_eq!(decrypted, payload, "XOR cipher must be its own inverse");
    }

    #[test]
    fn test_cipher_key_symbol_roundtrip() {
        for value in [0x0000u16, 0xFFFF, 0xAAAA, 0x5555, 0x1234, 0xDEAD] {
            let key = CipherKey::new(value);
            let mut symbols = [0u8; CIPHER_MASK_COUNT];
            for i in 0..CIPHER_MASK_COUNT {
                symbols[i] = key.symbol_for_mask(i);
                assert!(symbols[i] < MASK_QIM_MODULUS);
            }
            let reconstructed = CipherKey::from_symbols(&symbols);
            assert_eq!(
                reconstructed, key,
                "symbol round-trip failed for 0x{:04X}",
                value
            );
        }
    }

    #[test]
    fn test_generate_deterministic() {
        let seed = test_seed();
        let a = generate(&seed, 0);
        let b = generate(&seed, 0);
        assert_eq!(a, b, "same seed + packet_number must produce same cipher");
    }

    #[test]
    fn test_generate_varies_by_packet() {
        let seed = test_seed();
        let a = generate(&seed, 0);
        let b = generate(&seed, 1);
        // Not guaranteed to differ, but astronomically unlikely to match
        assert_ne!(a, b, "different packet numbers should produce different ciphers");
    }

    #[test]
    fn test_embed_extract_masks_roundtrip() {
        let config = QimConfig::new(0.01).unwrap();

        for cipher_val in [0x0000u16, 0xFFFF, 0xAAAA, 0x5555, 0x1234, 0xDEAD] {
            let cipher = CipherKey::new(cipher_val);
            let mask_indices: [usize; 8] = [100, 200, 300, 400, 500, 600, 700, 800];
            let mut samples = vec![0.5f64; 1000];

            embed_into_masks(&mut samples, &mask_indices, cipher, &config);
            let extracted = extract_from_masks(&samples, &mask_indices, &config);

            assert_eq!(
                extracted, cipher,
                "mask round-trip failed for 0x{:04X}: got {}",
                cipher_val, extracted
            );
        }
    }

    #[test]
    fn test_embed_extract_magnitudes_roundtrip() {
        let delta = 0.05;

        for cipher_val in [0x0000u16, 0xFFFF, 0xAAAA, 0x5555] {
            let cipher = CipherKey::new(cipher_val);
            let mask_indices: [usize; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
            let mut mags = vec![5.0f64; 100];

            embed_into_magnitudes(&mut mags, &mask_indices, cipher, delta);
            let extracted = extract_from_magnitudes(&mags, &mask_indices, delta);

            assert_eq!(
                extracted, cipher,
                "magnitude round-trip failed for 0x{:04X}",
                cipher_val
            );
        }
    }

    #[test]
    fn test_symbol_embed_extract_sweep() {
        let delta = 0.01;
        for m in [2u8, 4, 8] {
            for symbol in 0..m {
                for sample_i in -100..=100 {
                    let sample = sample_i as f64 / 100.0;
                    let embedded = embed_symbol(sample, symbol, m, delta);
                    let extracted = extract_symbol(embedded, m, delta);
                    assert_eq!(
                        extracted, symbol,
                        "m={}, symbol={}, sample={}",
                        m, symbol, sample
                    );
                }
            }
        }
    }

    #[test]
    fn test_nonneg_symbol_never_negative() {
        let delta = 0.05;
        for symbol in 0..4u8 {
            for mag_i in 0..=200 {
                let mag = mag_i as f64 / 100.0;
                let embedded = embed_symbol_nonneg(mag, symbol, 4, delta);
                assert!(
                    embedded >= 0.0,
                    "non-negative QIM produced negative: {} for mag={}, symbol={}",
                    embedded, mag, symbol
                );
                let extracted = extract_symbol_nonneg(embedded, 4, delta);
                assert_eq!(extracted, symbol);
            }
        }
    }
}
