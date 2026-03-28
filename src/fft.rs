//! FFT/IFFT operations via C math library.
//!
//! Provides safe Rust wrappers around the C radix-2 Cooley-Tukey FFT
//! implementation. All functions validate inputs and return Results
//! rather than exposing raw C error codes.

use std::fmt;

/// FFT-related errors.
#[derive(Debug, Clone)]
pub enum FftError {
    /// Input length is not a power of 2.
    NotPowerOfTwo(usize),
    /// Mismatched real/imaginary array lengths.
    LengthMismatch { re_len: usize, im_len: usize },
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FftError::NotPowerOfTwo(n) => write!(f, "FFT length {} is not a power of 2", n),
            FftError::LengthMismatch { re_len, im_len } => {
                write!(
                    f,
                    "real/imag length mismatch: re={}, im={}",
                    re_len, im_len
                )
            }
        }
    }
}

impl std::error::Error for FftError {}

// C FFI declarations
unsafe extern "C" {
    fn stenowav_fft(re: *mut f64, im: *mut f64, n: usize) -> i32;
    fn stenowav_ifft(re: *mut f64, im: *mut f64, n: usize) -> i32;
    fn stenowav_rms(samples: *const f64, n: usize) -> f64;
    fn stenowav_complex_mag(re: f64, im: f64) -> f64;
    fn stenowav_complex_phase(re: f64, im: f64) -> f64;
}

/// Check that n is a power of 2.
fn check_power_of_two(n: usize) -> Result<(), FftError> {
    if n == 0 || (n & (n - 1)) != 0 {
        Err(FftError::NotPowerOfTwo(n))
    } else {
        Ok(())
    }
}

/// In-place forward FFT.
///
/// `re` and `im` must have equal length, and that length must be a power of 2.
/// On return, they contain the frequency-domain representation.
pub fn fft(re: &mut [f64], im: &mut [f64]) -> Result<(), FftError> {
    if re.len() != im.len() {
        return Err(FftError::LengthMismatch {
            re_len: re.len(),
            im_len: im.len(),
        });
    }
    let n = re.len();
    if n <= 1 {
        return Ok(());
    }
    check_power_of_two(n)?;

    let ret = unsafe { stenowav_fft(re.as_mut_ptr(), im.as_mut_ptr(), n) };
    assert_eq!(ret, 0, "C FFT returned error after validation");
    Ok(())
}

/// In-place inverse FFT.
///
/// `re` and `im` must have equal length, and that length must be a power of 2.
/// On return, they contain the time-domain signal (already scaled by 1/n).
pub fn ifft(re: &mut [f64], im: &mut [f64]) -> Result<(), FftError> {
    if re.len() != im.len() {
        return Err(FftError::LengthMismatch {
            re_len: re.len(),
            im_len: im.len(),
        });
    }
    let n = re.len();
    if n <= 1 {
        return Ok(());
    }
    check_power_of_two(n)?;

    let ret = unsafe { stenowav_ifft(re.as_mut_ptr(), im.as_mut_ptr(), n) };
    assert_eq!(ret, 0, "C IFFT returned error after validation");
    Ok(())
}

/// Compute RMS of a signal segment.
pub fn rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    unsafe { stenowav_rms(samples.as_ptr(), samples.len()) }
}

/// Magnitude of a complex value.
pub fn complex_mag(re: f64, im: f64) -> f64 {
    unsafe { stenowav_complex_mag(re, im) }
}

/// Phase (argument) of a complex value.
pub fn complex_phase(re: f64, im: f64) -> f64 {
    unsafe { stenowav_complex_phase(re, im) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_fft_ifft_round_trip() {
        // Known signal: 4-sample impulse at index 0
        let original_re = vec![1.0, 0.0, 0.0, 0.0];
        let mut re = original_re.clone();
        let mut im = vec![0.0; 4];

        // Forward FFT of impulse -> all ones
        fft(&mut re, &mut im).unwrap();
        for &val in &re {
            assert!((val - 1.0).abs() < EPSILON, "FFT of impulse should be all 1s, got {}", val);
        }
        for &val in &im {
            assert!(val.abs() < EPSILON, "FFT of impulse should have zero imaginary");
        }

        // Inverse FFT should recover original
        ifft(&mut re, &mut im).unwrap();
        for i in 0..4 {
            assert!(
                (re[i] - original_re[i]).abs() < EPSILON,
                "IFFT should recover original at index {}: got {} expected {}",
                i, re[i], original_re[i]
            );
        }
    }

    #[test]
    fn test_fft_ifft_sine_wave() {
        let n = 64;
        let mut re: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let original = re.clone();
        let mut im = vec![0.0; n];

        fft(&mut re, &mut im).unwrap();
        ifft(&mut re, &mut im).unwrap();

        for i in 0..n {
            assert!(
                (re[i] - original[i]).abs() < EPSILON,
                "Round-trip failed at index {}: got {} expected {}",
                i, re[i], original[i]
            );
        }
    }

    #[test]
    fn test_fft_not_power_of_two() {
        let mut re = vec![0.0; 6];
        let mut im = vec![0.0; 6];
        assert!(fft(&mut re, &mut im).is_err());
    }

    #[test]
    fn test_fft_length_mismatch() {
        let mut re = vec![0.0; 4];
        let mut im = vec![0.0; 8];
        assert!(fft(&mut re, &mut im).is_err());
    }

    #[test]
    fn test_rms() {
        // RMS of constant signal = absolute value of that constant
        let signal = vec![3.0; 100];
        let r = rms(&signal);
        assert!((r - 3.0).abs() < EPSILON);

        // RMS of empty = 0
        assert_eq!(rms(&[]), 0.0);

        // RMS of [-1, 1] = 1
        let signal2 = vec![-1.0, 1.0];
        assert!((rms(&signal2) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_mag_phase() {
        // (3, 4) -> mag 5, phase atan2(4, 3)
        let m = complex_mag(3.0, 4.0);
        assert!((m - 5.0).abs() < EPSILON);

        let p = complex_phase(0.0, 1.0);
        assert!((p - std::f64::consts::FRAC_PI_2).abs() < EPSILON);
    }
}
