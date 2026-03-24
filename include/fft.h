#ifndef FFT_H
#define FFT_H

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>

// In-place radix-2 Cooley-Tukey FFT/IFFT for real-valued audio signals.

class FFT {
public:
    using Complex = std::complex<double>;

    // Forward FFT (in-place). buf.size() must be power of 2.
    static void forward(std::vector<Complex>& buf) {
        size_t N = buf.size();
        if (N <= 1) return;

        // Bit-reversal permutation
        for (size_t i = 1, j = 0; i < N; i++) {
            size_t bit = N >> 1;
            for (; j & bit; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) std::swap(buf[i], buf[j]);
        }

        // Butterfly stages
        for (size_t len = 2; len <= N; len <<= 1) {
            double angle = -2.0 * M_PI / static_cast<double>(len);
            Complex wlen(std::cos(angle), std::sin(angle));
            for (size_t i = 0; i < N; i += len) {
                Complex w(1.0, 0.0);
                for (size_t j = 0; j < len / 2; j++) {
                    Complex u = buf[i + j];
                    Complex v = buf[i + j + len / 2] * w;
                    buf[i + j] = u + v;
                    buf[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    // Inverse FFT (in-place). Divides by N.
    static void inverse(std::vector<Complex>& buf) {
        size_t N = buf.size();
        // Conjugate
        for (auto& c : buf) c = std::conj(c);
        forward(buf);
        for (auto& c : buf) c = std::conj(c) / static_cast<double>(N);
    }

    // Next power of 2 >= n
    static size_t next_pow2(size_t n) {
        size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }
};

#endif
