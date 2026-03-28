//! Integration tests that verify cross-module behaviour.
//!
//! These test the full pipeline: load WAV -> QIM embed -> write WAV -> read WAV -> QIM extract.

use stenowav::dither::{self, DitherConfig};
use stenowav::embed::{self, EmbedConfig, EmbedDomain};
use stenowav::fft;
use stenowav::packet;
use stenowav::qim::{self, QimConfig};
use stenowav::wav::{self, WavData};

/// Full pipeline: embed a message into a WAV file, write it, read it back, extract the message.
#[test]
fn test_wav_qim_round_trip() {
    let delta = 0.005;
    let config = QimConfig::new(delta).unwrap();

    // Create a synthetic stereo signal (440Hz + 880Hz, 0.5 seconds)
    let sample_rate = 44100u32;
    let duration_samples = sample_rate as usize / 2;
    let left: Vec<f64> = (0..duration_samples)
        .map(|i| 0.8 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin())
        .collect();
    let right: Vec<f64> = (0..duration_samples)
        .map(|i| 0.6 * (2.0 * std::f64::consts::PI * 880.0 * i as f64 / sample_rate as f64).sin())
        .collect();

    let mut audio = WavData {
        sample_rate,
        bits_per_sample: 16,
        channels: vec![left, right],
    };

    // Message to embed: "StenoWav"
    let message = b"StenoWav";

    // Embed into left channel starting at sample 1000
    let start_idx = 1000;
    for (byte_idx, &byte) in message.iter().enumerate() {
        let offset = start_idx + byte_idx * 8;
        let samples = &mut audio.channels[0][offset..offset + 8];
        qim::embed_byte(samples, byte, &config).unwrap();
    }

    // Write watermarked audio
    let tmp_path = "/tmp/stenowav_integration_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();

    // Read it back
    let loaded = wav::read_wav(tmp_path).unwrap();
    assert_eq!(loaded.sample_rate, sample_rate);
    assert_eq!(loaded.num_channels(), 2);

    // Extract the message from the loaded audio
    // Note: there will be quantisation noise from i16 conversion, so QIM delta
    // must be large enough to survive it. delta=0.005 >> 1/32768 ≈ 0.00003, so safe.
    let mut extracted = Vec::new();
    for byte_idx in 0..message.len() {
        let offset = start_idx + byte_idx * 8;
        let samples = &loaded.channels[0][offset..offset + 8];
        extracted.push(qim::extract_byte(samples, &config));
    }

    assert_eq!(
        &extracted,
        message,
        "extracted {:?} != original {:?}",
        String::from_utf8_lossy(&extracted),
        String::from_utf8_lossy(message)
    );

    // Cleanup
    let _ = std::fs::remove_file(tmp_path);
}

/// Verify FFT/IFFT preserves signal properties after QIM embedding.
#[test]
fn test_fft_after_qim_embedding() {
    let config = QimConfig::new(0.01).unwrap();

    // Create a signal (power of 2 length for FFT)
    let n = 256;
    let mut signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
        .collect();
    let original = signal.clone();

    // Embed some bits
    let bits: Vec<u8> = (0..16).map(|i| i % 2).collect();
    qim::embed_bits(&mut signal[100..116], &bits, &config).unwrap();

    // FFT the original and modified signals
    let mut orig_re = original.clone();
    let mut orig_im = vec![0.0; n];
    fft::fft(&mut orig_re, &mut orig_im).unwrap();

    let mut mod_re = signal.clone();
    let mut mod_im = vec![0.0; n];
    fft::fft(&mut mod_re, &mut mod_im).unwrap();

    // The spectral difference should be small (QIM introduces minimal distortion)
    let mut max_spectral_diff = 0.0f64;
    for i in 0..n {
        let orig_mag = fft::complex_mag(orig_re[i], orig_im[i]);
        let mod_mag = fft::complex_mag(mod_re[i], mod_im[i]);
        max_spectral_diff = max_spectral_diff.max((orig_mag - mod_mag).abs());
    }

    // With delta=0.01, max per-sample distortion is 0.01, spread across 16 samples.
    // Spectral leakage means max_spectral_diff should be bounded but not zero.
    assert!(
        max_spectral_diff < 1.0,
        "spectral distortion too large: {}",
        max_spectral_diff
    );
}

/// Test that QIM survives the WAV quantisation for the real audio file.
#[test]
fn test_real_audio_qim_embed_extract() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = QimConfig::new(0.005).unwrap();

    // Embed "HELLO" at a known offset
    let message = b"HELLO";
    let start = 50000;
    for (i, &byte) in message.iter().enumerate() {
        let offset = start + i * 8;
        qim::embed_byte(&mut audio.channels[0][offset..offset + 8], byte, &config).unwrap();
    }

    // Write and read back
    let tmp_path = "/tmp/stenowav_real_audio_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    // Extract
    let mut extracted = Vec::new();
    for i in 0..message.len() {
        let offset = start + i * 8;
        extracted.push(qim::extract_byte(&loaded.channels[0][offset..offset + 8], &config));
    }

    assert_eq!(&extracted, message);

    // Verify audio isn't significantly altered - compute max sample difference
    let original = wav::read_wav(wav_path).unwrap();
    let mut max_diff = 0.0f64;
    for i in 0..original.num_samples().min(loaded.num_samples()) {
        let diff = (original.channels[0][i] - loaded.channels[0][i]).abs();
        max_diff = max_diff.max(diff);
    }
    // Max diff should be bounded by delta + quantisation error
    assert!(
        max_diff < 0.01,
        "max sample difference {} is too large",
        max_diff
    );

    let _ = std::fs::remove_file(tmp_path);
}

/// Verify RMS computation on real audio segments.
#[test]
fn test_rms_on_audio_segments() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let audio = wav::read_wav(wav_path).unwrap();
    let samples = &audio.channels[0];

    // Compute RMS on different segments
    let segment_size = 4096;
    let mut rms_values = Vec::new();
    for chunk in samples.chunks(segment_size) {
        let r = fft::rms(chunk);
        assert!(r >= 0.0, "RMS must be non-negative");
        assert!(r.is_finite(), "RMS must be finite");
        rms_values.push(r);
    }

    // There should be variation in RMS across a real music signal
    let min_rms = rms_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rms = rms_values.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        max_rms > min_rms,
        "expected RMS variation in real audio, got min={} max={}",
        min_rms, max_rms
    );
}

/// Full packet pipeline: encode a message via packets into real audio, write WAV, read back, decode.
#[test]
fn test_packet_pipeline_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = QimConfig::new(0.005).unwrap();

    let seed: [u8; 32] = [
        0x53, 0x74, 0x65, 0x6E, 0x6F, 0x57, 0x61, 0x76, // "StenoWav"
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    ];

    let message = b"Steganography is the art of hiding in plain sight!";

    // Encode into left channel
    let num_packets = packet::encode_stream(
        &mut audio.channels[0],
        &seed,
        message,
        &config,
    )
    .unwrap();

    // Write watermarked audio and read back
    let tmp_path = "/tmp/stenowav_packet_pipeline_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    // Decode from the loaded (quantised) audio
    let decoded = packet::decode_stream(
        &loaded.channels[0],
        &seed,
        num_packets,
        Some(message.len()),
        &config,
    )
    .unwrap();

    assert_eq!(
        &decoded,
        message,
        "packet pipeline failed on real audio: got {:?}",
        String::from_utf8_lossy(&decoded)
    );

    let _ = std::fs::remove_file(tmp_path);
}

/// Verify that packet encoding with a wrong seed produces garbage.
#[test]
fn test_packet_wrong_seed_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = QimConfig::new(0.005).unwrap();

    let seed: [u8; 32] = [0xAA; 32];
    let wrong_seed: [u8; 32] = [0xBB; 32];

    let message = b"Secret data";
    let num_packets = packet::encode_stream(
        &mut audio.channels[0],
        &seed,
        message,
        &config,
    )
    .unwrap();

    // Try decoding with wrong seed
    let decoded = packet::decode_stream(
        &audio.channels[0],
        &wrong_seed,
        num_packets,
        Some(message.len()),
        &config,
    )
    .unwrap();

    assert_ne!(
        &decoded, message,
        "wrong seed should not produce correct output"
    );
}

// ── Dual-domain embedding integration tests ──────────────────────

fn steno_seed() -> [u8; 32] {
    [
        0x53, 0x74, 0x65, 0x6E, 0x6F, 0x57, 0x61, 0x76, // "StenoWav"
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    ]
}

/// Time-domain RMS-adaptive embedding on real audio, surviving WAV i16 quantisation.
#[test]
fn test_embed_time_domain_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        ..Default::default()
    };

    let message = b"Time-domain steganography on My Own Summer!";

    // Encode into left channel
    let meta = embed::encode_time(&mut audio.channels[0], message, &config).unwrap();

    // Write watermarked audio and read back (i16 quantisation round-trip)
    let tmp_path = "/tmp/stenowav_embed_time_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    // Decode from quantised audio
    let decoded = embed::decode_time(&loaded.channels[0], &meta, &config).unwrap();

    assert_eq!(
        &decoded, message,
        "time-domain embed/decode failed on real audio: got {:?}",
        String::from_utf8_lossy(&decoded)
    );

    let _ = std::fs::remove_file(tmp_path);
}

/// Frequency-domain embedding on real audio, surviving WAV i16 quantisation.
#[test]
fn test_embed_freq_domain_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Frequency,
        seed: steno_seed(),
        ..Default::default()
    };

    let message = b"Freq-domain secrets in music!";

    let meta = embed::encode_freq(&mut audio.channels[0], message, &config).unwrap();

    let tmp_path = "/tmp/stenowav_embed_freq_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    let decoded = embed::decode_freq(&loaded.channels[0], &meta, &config).unwrap();

    assert_eq!(
        &decoded, message,
        "freq-domain embed/decode failed on real audio: got {:?}",
        String::from_utf8_lossy(&decoded)
    );

    let _ = std::fs::remove_file(tmp_path);
}

// ── Priority 4: Full pipeline (header-embedded) round-trip tests ──

/// Full encode→decode pipeline on real audio, time domain, WAV round-trip.
/// The decoder uses only the watermarked signal + seed — no EncodeMeta needed.
#[test]
fn test_full_pipeline_time_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        ..Default::default()
    };

    let message = b"Full pipeline time-domain on real audio!";

    embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

    // WAV i16 quantisation round-trip
    let tmp_path = "/tmp/stenowav_full_time_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    // Decode without EncodeMeta — just signal + seed + config
    let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
    assert_eq!(
        &decoded, message,
        "full time-domain pipeline failed on real audio: got {:?}",
        String::from_utf8_lossy(&decoded)
    );

    let _ = std::fs::remove_file(tmp_path);
}

/// Full encode→decode pipeline on real audio, frequency domain, WAV round-trip.
#[test]
fn test_full_pipeline_freq_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Frequency,
        seed: steno_seed(),
        ..Default::default()
    };

    let message = b"Freq full pipeline real audio!";

    embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

    let tmp_path = "/tmp/stenowav_full_freq_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
    assert_eq!(
        &decoded, message,
        "full freq-domain pipeline failed on real audio: got {:?}",
        String::from_utf8_lossy(&decoded)
    );

    let _ = std::fs::remove_file(tmp_path);
}

/// Wrong seed should fail header validation on real audio.
#[test]
fn test_full_pipeline_wrong_seed_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        ..Default::default()
    };

    embed::encode_full(&mut audio.channels[0], b"Secret", &config).unwrap();

    let tmp_path = "/tmp/stenowav_full_wrong_seed_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    let mut wrong_config = config.clone();
    wrong_config.seed[0] ^= 0xFF;
    let result = embed::decode_full(&loaded.channels[0], &wrong_config);
    assert!(result.is_err(), "wrong seed should fail header check");

    let _ = std::fs::remove_file(tmp_path);
}

/// Parameter sweep: test multiple deltas on real audio time-domain round-trip.
#[test]
fn test_full_pipeline_delta_sweep_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    for &delta in &[0.003, 0.005, 0.01] {
        let mut audio = wav::read_wav(wav_path).unwrap();
        let config = EmbedConfig {
            domain: EmbedDomain::Time,
            seed: steno_seed(),
            time_delta: delta,
            ..Default::default()
        };

        let message = b"Delta sweep!";
        embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

        let tmp_path = "/tmp/stenowav_delta_sweep_test.wav";
        wav::write_wav(tmp_path, &audio).unwrap();
        let loaded = wav::read_wav(tmp_path).unwrap();

        let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
        assert_eq!(
            &decoded, message,
            "delta sweep failed at delta={}: got {:?}",
            delta,
            String::from_utf8_lossy(&decoded)
        );

        let _ = std::fs::remove_file(tmp_path);
    }
}

/// Multiple message sizes through the full pipeline on real audio.
#[test]
fn test_full_pipeline_various_sizes_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        ..Default::default()
    };

    for payload in [
        &b""[..],
        &b"A"[..],
        &b"AB"[..],
        &b"Hello StenoWav!"[..],
        &b"Steganography hides data in plain sight"[..],
    ] {
        let mut audio = wav::read_wav(wav_path).unwrap();
        embed::encode_full(&mut audio.channels[0], payload, &config).unwrap();

        let tmp_path = "/tmp/stenowav_sizes_test.wav";
        wav::write_wav(tmp_path, &audio).unwrap();
        let loaded = wav::read_wav(tmp_path).unwrap();

        let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
        assert_eq!(
            &decoded, payload,
            "full pipeline size test failed for {:?}",
            String::from_utf8_lossy(payload)
        );

        let _ = std::fs::remove_file(tmp_path);
    }
}

/// Verify audio quality: max sample difference after time-domain embedding is small.
#[test]
fn test_embed_time_domain_audio_quality() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let original = wav::read_wav(wav_path).unwrap();
    let mut audio = original.clone();
    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        ..Default::default()
    };

    let message = b"Quality check message";
    embed::encode_time(&mut audio.channels[0], message, &config).unwrap();

    // Max per-sample distortion should be bounded by delta + quantisation
    let mut max_diff = 0.0f64;
    for i in 0..original.num_samples() {
        let diff = (original.channels[0][i] - audio.channels[0][i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(
        max_diff < 0.01,
        "time-domain embedding distortion {} exceeds bound",
        max_diff
    );
}

// ══════════════════════════════════════════════════════════════════
// Dither integration tests
// ══════════════════════════════════════════════════════════════════

/// Full pipeline time-domain with dither enabled — data must survive.
#[test]
fn test_full_pipeline_time_dither_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        dither: DitherConfig {
            enabled: true,
            intensity: 2.0 / 32768.0, // ~2 LSBs
        },
        ..Default::default()
    };

    let message = b"Dither time-domain on real audio!";

    embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

    // Write and read back through i16 quantisation
    let tmp_path = "/tmp/stenowav_dither_time_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
    assert_eq!(
        &decoded, message,
        "dither should not corrupt payload: got {:?}",
        String::from_utf8_lossy(&decoded)
    );
}

/// Full pipeline frequency-domain with dither enabled — data must survive.
#[test]
fn test_full_pipeline_freq_dither_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let config = EmbedConfig {
        domain: EmbedDomain::Frequency,
        seed: steno_seed(),
        dither: DitherConfig {
            enabled: true,
            intensity: 2.0 / 32768.0,
        },
        ..Default::default()
    };

    let message = b"Dither freq-domain secrets!";

    embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

    let tmp_path = "/tmp/stenowav_dither_freq_test.wav";
    wav::write_wav(tmp_path, &audio).unwrap();
    let loaded = wav::read_wav(tmp_path).unwrap();

    let decoded = embed::decode_full(&loaded.channels[0], &config).unwrap();
    assert_eq!(
        &decoded, message,
        "freq dither should not corrupt payload: got {:?}",
        String::from_utf8_lossy(&decoded)
    );
}

/// Quality metrics on real audio with dither should show acceptable SNR.
#[test]
fn test_dither_quality_metrics_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let mut audio = wav::read_wav(wav_path).unwrap();
    let original = audio.channels[0].clone();

    let config = EmbedConfig {
        domain: EmbedDomain::Time,
        seed: steno_seed(),
        dither: DitherConfig {
            enabled: true,
            intensity: 2.0 / 32768.0,
        },
        ..Default::default()
    };

    let message = b"Quality metrics with dither!";
    embed::encode_full(&mut audio.channels[0], message, &config).unwrap();

    let metrics = dither::compute_quality(&original, &audio.channels[0]);

    // SNR should be > 30 dB for imperceptible watermarking
    assert!(
        metrics.snr_db > 30.0,
        "SNR {:.1} dB is too low for imperceptible watermarking",
        metrics.snr_db
    );

    // Max sample distortion should be small
    assert!(
        metrics.max_sample_diff < 0.01,
        "max sample diff {} exceeds perceptual bound",
        metrics.max_sample_diff
    );

    // Spectral difference should be bounded
    let spectral_diff = dither::max_spectral_diff(&original, &audio.channels[0]).unwrap();
    assert!(
        spectral_diff < 5.0,
        "max spectral diff {} is too large",
        spectral_diff
    );
}

/// Dither without embedding — pure dither produces negligible distortion.
#[test]
fn test_dither_only_quality_real_audio() {
    let wav_path = "/home/mecilh/Sandboxing/StenWav/sounds/MyOwnSummer.wav";
    if !std::path::Path::new(wav_path).exists() {
        return;
    }

    let audio = wav::read_wav(wav_path).unwrap();
    let mut dithered = audio.channels[0].clone();
    let original = audio.channels[0].clone();

    let config = DitherConfig {
        enabled: true,
        intensity: 2.0 / 32768.0,
    };

    // Apply dither with no protected indices (dithers everything)
    let protected = std::collections::HashSet::new();
    dither::apply_time_dither(&mut dithered, &protected, &steno_seed(), &config);

    let metrics = dither::compute_quality(&original, &dithered);

    // Pure dither at 2 LSBs should have very high SNR (> 50 dB)
    assert!(
        metrics.snr_db > 50.0,
        "pure dither SNR {:.1} dB is too low — dither should be inaudible",
        metrics.snr_db
    );

    // Max sample diff should be bounded by intensity
    assert!(
        metrics.max_sample_diff <= config.intensity + 1e-15,
        "max diff {} exceeds intensity {}",
        metrics.max_sample_diff,
        config.intensity
    );
}
