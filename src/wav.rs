//! WAV file reader and writer for 16-bit PCM audio.
//!
//! Supports reading and writing stereo/mono WAV files in the standard
//! RIFF WAVE format with PCM encoding. Samples are exposed as f64 in
//! the range [-1.0, 1.0] for processing, and converted back to i16
//! on write.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Errors that can occur during WAV operations.
#[derive(Debug)]
pub enum WavError {
    /// I/O error from the filesystem.
    Io(io::Error),
    /// The file is not a valid RIFF WAVE file.
    InvalidFormat(String),
    /// Unsupported audio format (we only handle PCM 16-bit).
    UnsupportedFormat(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::Io(e) => write!(f, "I/O error: {}", e),
            WavError::InvalidFormat(msg) => write!(f, "invalid WAV format: {}", msg),
            WavError::UnsupportedFormat(msg) => write!(f, "unsupported WAV format: {}", msg),
        }
    }
}

impl std::error::Error for WavError {}

impl From<io::Error> for WavError {
    fn from(e: io::Error) -> Self {
        WavError::Io(e)
    }
}

/// Represents a loaded WAV audio file.
///
/// Samples are stored per-channel as f64 in [-1.0, 1.0].
/// For stereo files, `channels[0]` is left and `channels[1]` is right.
#[derive(Debug, Clone)]
pub struct WavData {
    /// Sample rate in Hz (e.g., 44100).
    pub sample_rate: u32,
    /// Bits per sample (always 16 for now).
    pub bits_per_sample: u16,
    /// Per-channel sample data, normalised to [-1.0, 1.0].
    pub channels: Vec<Vec<f64>>,
}

impl WavData {
    /// Number of audio channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel.
    pub fn num_samples(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.num_samples() as f64 / self.sample_rate as f64
    }
}

// ── Reading ──────────────────────────────────────────────────────

/// Read a 16-bit PCM WAV file from the given path.
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<WavData, WavError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // RIFF header
    let riff_id = read_4bytes(&mut reader)?;
    if &riff_id != b"RIFF" {
        return Err(WavError::InvalidFormat("missing RIFF header".into()));
    }
    let _file_size = read_u32_le(&mut reader)?;
    let wave_id = read_4bytes(&mut reader)?;
    if &wave_id != b"WAVE" {
        return Err(WavError::InvalidFormat("missing WAVE identifier".into()));
    }

    // Parse chunks — we need "fmt " and "data"
    let mut sample_rate: u32 = 0;
    let mut num_channels: u16 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut audio_data: Vec<u8> = Vec::new();
    let mut found_fmt = false;
    let mut found_data = false;

    loop {
        let chunk_id = match read_4bytes(&mut reader) {
            Ok(id) => id,
            Err(_) => break, // End of file
        };
        let chunk_size = read_u32_le(&mut reader)?;

        if &chunk_id == b"fmt " {
            let audio_format = read_u16_le(&mut reader)?;
            if audio_format != 1 {
                return Err(WavError::UnsupportedFormat(format!(
                    "audio format {} (only PCM=1 supported)",
                    audio_format
                )));
            }
            num_channels = read_u16_le(&mut reader)?;
            sample_rate = read_u32_le(&mut reader)?;
            let _byte_rate = read_u32_le(&mut reader)?;
            let _block_align = read_u16_le(&mut reader)?;
            bits_per_sample = read_u16_le(&mut reader)?;

            if bits_per_sample != 16 {
                return Err(WavError::UnsupportedFormat(format!(
                    "{}-bit audio (only 16-bit supported)",
                    bits_per_sample
                )));
            }

            // Skip any extra bytes in the fmt chunk
            let fmt_read = 16u32;
            if chunk_size > fmt_read {
                skip_bytes(&mut reader, (chunk_size - fmt_read) as usize)?;
            }
            found_fmt = true;
        } else if &chunk_id == b"data" {
            audio_data = vec![0u8; chunk_size as usize];
            reader.read_exact(&mut audio_data)?;
            found_data = true;
        } else {
            // Skip unknown chunks
            skip_bytes(&mut reader, chunk_size as usize)?;
        }
    }

    if !found_fmt {
        return Err(WavError::InvalidFormat("missing fmt chunk".into()));
    }
    if !found_data {
        return Err(WavError::InvalidFormat("missing data chunk".into()));
    }

    // Deinterleave and convert i16 -> f64
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let block_align = num_channels as usize * bytes_per_sample;
    let num_frames = audio_data.len() / block_align;

    let mut channels: Vec<Vec<f64>> = (0..num_channels as usize)
        .map(|_| Vec::with_capacity(num_frames))
        .collect();

    for frame in 0..num_frames {
        for ch in 0..num_channels as usize {
            let offset = frame * block_align + ch * bytes_per_sample;
            let sample_i16 = i16::from_le_bytes([audio_data[offset], audio_data[offset + 1]]);
            // Normalise to [-1.0, 1.0]
            let sample_f64 = sample_i16 as f64 / 32768.0;
            channels[ch].push(sample_f64);
        }
    }

    Ok(WavData {
        sample_rate,
        bits_per_sample,
        channels,
    })
}

// ── Writing ──────────────────────────────────────────────────────

/// Write a WAV file from WavData to the given path.
///
/// Converts f64 samples back to 16-bit PCM with proper clamping.
pub fn write_wav<P: AsRef<Path>>(path: P, data: &WavData) -> Result<(), WavError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_channels = data.num_channels() as u16;
    let num_samples = data.num_samples();
    let bits_per_sample = 16u16;
    let bytes_per_sample = (bits_per_sample / 8) as u32;
    let block_align = num_channels as u32 * bytes_per_sample;
    let byte_rate = data.sample_rate * block_align;
    let data_size = num_samples as u32 * block_align;
    let file_size = 36 + data_size; // 36 = header bytes before data chunk payload

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // chunk size
    writer.write_all(&1u16.to_le_bytes())?; // PCM format
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&data.sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&(block_align as u16).to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // Interleave channels and convert f64 -> i16
    for frame in 0..num_samples {
        for ch in 0..num_channels as usize {
            let sample_f64 = data.channels[ch][frame];
            // Clamp to [-1.0, 1.0] then scale to i16 range
            let clamped = sample_f64.clamp(-1.0, 1.0);
            // Use 32768.0 for both encode and decode to keep the conversion symmetric.
            // Clamp the result to i16 range to handle the +1.0 edge case.
            let scaled = (clamped * 32768.0).round();
            let sample_i16 = scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            writer.write_all(&sample_i16.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

// ── Helper I/O functions ────────────────────────────────────────

fn read_4bytes<R: Read>(reader: &mut R) -> Result<[u8; 4], WavError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32, WavError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16, WavError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn skip_bytes<R: Read>(reader: &mut R, count: usize) -> Result<(), WavError> {
    let mut remaining = count;
    let mut buf = [0u8; 1024];
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        reader.read_exact(&mut buf[..to_read])?;
        remaining -= to_read;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_write_read_round_trip() {
        let tmp_path = PathBuf::from("/tmp/stenowav_test_round_trip.wav");

        // Create a simple stereo signal
        let n = 4410; // 0.1 seconds at 44100Hz
        let left: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let right: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 880.0 * i as f64 / 44100.0).sin())
            .collect();

        let original = WavData {
            sample_rate: 44100,
            bits_per_sample: 16,
            channels: vec![left, right],
        };

        // Write then read back
        write_wav(&tmp_path, &original).unwrap();
        let loaded = read_wav(&tmp_path).unwrap();

        assert_eq!(loaded.sample_rate, 44100);
        assert_eq!(loaded.num_channels(), 2);
        assert_eq!(loaded.num_samples(), n);

        // Check samples are close (quantisation error from f64->i16->f64)
        // Worst case is 1 LSB = 1/32768 ≈ 3.05e-5.
        let max_quant_error = 1.0 / 32768.0 + 1e-10;
        for ch in 0..2 {
            for i in 0..n {
                let diff = (loaded.channels[ch][i] - original.channels[ch][i]).abs();
                assert!(
                    diff < max_quant_error,
                    "ch={} i={}: diff {} exceeds quantisation error",
                    ch, i, diff
                );
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_read_real_wav_file() {
        // Test with the actual WAV file in the project
        let wav_path = PathBuf::from("/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav");
        if !wav_path.exists() {
            return; // Skip if file not available
        }

        let data = read_wav(&wav_path).unwrap();
        assert_eq!(data.sample_rate, 44100);
        assert_eq!(data.num_channels(), 2); // Stereo
        assert_eq!(data.bits_per_sample, 16);
        assert!(data.num_samples() > 0);
        assert!(data.duration_secs() > 0.0);
    }

    #[test]
    fn test_mono_round_trip() {
        let tmp_path = PathBuf::from("/tmp/stenowav_test_mono.wav");

        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 / 100.0).sin()).collect();
        let data = WavData {
            sample_rate: 22050,
            bits_per_sample: 16,
            channels: vec![signal],
        };

        write_wav(&tmp_path, &data).unwrap();
        let loaded = read_wav(&tmp_path).unwrap();

        assert_eq!(loaded.num_channels(), 1);
        assert_eq!(loaded.num_samples(), 1000);
        assert_eq!(loaded.sample_rate, 22050);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
