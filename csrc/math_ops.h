#ifndef STENOWAV_MATH_OPS_H
#define STENOWAV_MATH_OPS_H

#include <stddef.h>

/*
 * Complex number representation for FFT operations.
 * Stored as interleaved real/imaginary pairs in the arrays,
 * but this struct is used for intermediate computations.
 */
typedef struct {
    double re;
    double im;
} complex_t;

/*
 * In-place radix-2 Cooley-Tukey FFT.
 *
 * Parameters:
 *   re[]  - real parts of the input/output (length n)
 *   im[]  - imaginary parts of the input/output (length n)
 *   n     - number of samples, MUST be a power of 2
 *
 * On return, re[] and im[] contain the frequency-domain representation.
 * Returns 0 on success, -1 if n is not a power of 2.
 */
int stenowav_fft(double *re, double *im, size_t n);

/*
 * In-place inverse FFT.
 *
 * Same interface as stenowav_fft. On return, re[] and im[] contain
 * the time-domain signal (already scaled by 1/n).
 * Returns 0 on success, -1 if n is not a power of 2.
 */
int stenowav_ifft(double *re, double *im, size_t n);

/*
 * Compute RMS (root mean square) of a real-valued signal segment.
 *
 * Parameters:
 *   samples[] - input samples
 *   n         - number of samples
 *
 * Returns the RMS value. Returns 0.0 if n == 0.
 */
double stenowav_rms(const double *samples, size_t n);

/*
 * Compute magnitude (absolute value) of a complex number.
 */
double stenowav_complex_mag(double re, double im);

/*
 * Compute phase (argument) of a complex number.
 */
double stenowav_complex_phase(double re, double im);

#endif /* STENOWAV_MATH_OPS_H */
