#include "math_ops.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Check if n is a power of 2 */
static int is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* Bit-reversal permutation for in-place FFT */
static void bit_reverse(double *re, double *im, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n - 1; i++) {
        if (i < j) {
            double tmp_re = re[i];
            double tmp_im = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tmp_re;
            im[j] = tmp_im;
        }
        size_t k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

/*
 * In-place radix-2 Cooley-Tukey FFT.
 *
 * The algorithm:
 *   1. Bit-reverse the input indices
 *   2. Butterfly passes at each stage, doubling the sub-FFT size
 *   3. Twiddle factors: W_N^k = exp(-2*pi*i*k/N)
 */
int stenowav_fft(double *re, double *im, size_t n) {
    if (!is_power_of_two(n)) {
        return -1;
    }
    if (n <= 1) {
        return 0;
    }

    bit_reverse(re, im, n);

    /* Butterfly passes */
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / (double)len;
        double w_re = cos(angle);
        double w_im = sin(angle);

        for (size_t i = 0; i < n; i += len) {
            double wn_re = 1.0;
            double wn_im = 0.0;

            for (size_t j = 0; j < len / 2; j++) {
                size_t even = i + j;
                size_t odd  = i + j + len / 2;

                /* Butterfly: twiddle * odd element */
                double t_re = wn_re * re[odd] - wn_im * im[odd];
                double t_im = wn_re * im[odd] + wn_im * re[odd];

                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] = re[even] + t_re;
                im[even] = im[even] + t_im;

                /* Advance twiddle factor */
                double new_wn_re = wn_re * w_re - wn_im * w_im;
                double new_wn_im = wn_re * w_im + wn_im * w_re;
                wn_re = new_wn_re;
                wn_im = new_wn_im;
            }
        }
    }

    return 0;
}

/*
 * Inverse FFT: conjugate input, forward FFT, conjugate output, scale by 1/n.
 */
int stenowav_ifft(double *re, double *im, size_t n) {
    if (!is_power_of_two(n)) {
        return -1;
    }
    if (n <= 1) {
        return 0;
    }

    /* Conjugate the input */
    for (size_t i = 0; i < n; i++) {
        im[i] = -im[i];
    }

    /* Forward FFT */
    int ret = stenowav_fft(re, im, n);
    if (ret != 0) {
        return ret;
    }

    /* Conjugate and scale by 1/n */
    double inv_n = 1.0 / (double)n;
    for (size_t i = 0; i < n; i++) {
        re[i] *= inv_n;
        im[i] = -im[i] * inv_n;
    }

    return 0;
}

double stenowav_rms(const double *samples, size_t n) {
    if (n == 0) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_sq += samples[i] * samples[i];
    }

    return sqrt(sum_sq / (double)n);
}

double stenowav_complex_mag(double re, double im) {
    return sqrt(re * re + im * im);
}

double stenowav_complex_phase(double re, double im) {
    return atan2(im, re);
}
