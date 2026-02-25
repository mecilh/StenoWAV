#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ── WAV header (44 bytes, PCM) ──────────────────────────────────────────────
#pragma pack(push, 1)
struct WavHeader {
    char     riff_id[4];       // "RIFF"
    uint32_t file_size;        // file size - 8
    char     wave_id[4];       // "WAVE"
    char     fmt_id[4];        // "fmt "
    uint32_t fmt_size;         // 16 for PCM
    uint16_t audio_format;     // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

// ── Cooley-Tukey FFT (radix-2, in-place) ────────────────────────────────────
using Complex = std::complex<double>;

static void fft(std::vector<Complex>& x) {
    const size_t n = x.size();
    if (n <= 1) return;

    // bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // butterfly
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / static_cast<double>(len);
        Complex wn(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = x[i + j];
                Complex t = w * x[i + j + len / 2];
                x[i + j]           = u + t;
                x[i + j + len / 2] = u - t;
                w *= wn;
            }
        }
    }
}

// ── Utilities ───────────────────────────────────────────────────────────────
static size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static bool read_wav(const std::string& path,
                     WavHeader& header,
                     std::vector<double>& samples_left,
                     std::vector<double>& samples_right) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Errore: impossibile aprire \"" << path << "\"\n";
        return false;
    }

    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (std::strncmp(header.riff_id, "RIFF", 4) != 0 ||
        std::strncmp(header.wave_id, "WAVE", 4) != 0) {
        std::cerr << "Errore: non e' un file WAV valido\n";
        return false;
    }
    if (header.audio_format != 1) {
        std::cerr << "Errore: supportato solo PCM (format=1), trovato "
                  << header.audio_format << "\n";
        return false;
    }

    // skip to "data" chunk
    char chunk_id[4];
    uint32_t chunk_size;
    while (file.read(chunk_id, 4)) {
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::strncmp(chunk_id, "data", 4) == 0) break;
        file.seekg(chunk_size, std::ios::cur);
    }

    const int bytes_per_sample = header.bits_per_sample / 8;
    const size_t total_samples = chunk_size / bytes_per_sample;
    const size_t frames = total_samples / header.num_channels;

    samples_left.resize(frames);
    if (header.num_channels == 2) samples_right.resize(frames);

    const double max_val = (1 << (header.bits_per_sample - 1)) - 1;

    for (size_t i = 0; i < frames; ++i) {
        for (uint16_t ch = 0; ch < header.num_channels; ++ch) {
            int32_t raw = 0;
            file.read(reinterpret_cast<char*>(&raw), bytes_per_sample);
            // sign-extend for 16-bit
            if (bytes_per_sample == 2)
                raw = static_cast<int16_t>(raw & 0xFFFF);
            else if (bytes_per_sample == 3 && (raw & 0x800000))
                raw |= ~0xFFFFFF;

            double normalized = static_cast<double>(raw) / max_val;
            if (ch == 0) samples_left[i] = normalized;
            else         samples_right[i] = normalized;
        }
    }

    return true;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <file.wav> [fft_size]\n";
        return 1;
    }

    const std::string wav_path = argv[1];
    const size_t fft_size = (argc >= 3) ? std::stoul(argv[2]) : 1024;

    WavHeader header{};
    std::vector<double> left, right;

    if (!read_wav(wav_path, header, left, right))
        return 1;

    std::cout << "=== WAV Info ===\n"
              << "  Sample rate:      " << header.sample_rate << " Hz\n"
              << "  Channels:         " << header.num_channels << "\n"
              << "  Bits per sample:  " << header.bits_per_sample << "\n"
              << "  Total frames:     " << left.size() << "\n"
              << "  Durata:           "
              << static_cast<double>(left.size()) / header.sample_rate
              << " s\n\n";

    // FFT on the first segment of the left channel
    const size_t n = next_power_of_2(fft_size);
    const size_t seg_len = std::min(n, left.size());

    // apply Hanning window + copy into complex buffer
    std::vector<Complex> buf(n, Complex(0.0));
    for (size_t i = 0; i < seg_len; ++i) {
        double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (seg_len - 1)));
        buf[i] = Complex(left[i] * w, 0.0);
    }

    fft(buf);

    // print magnitude spectrum (only positive frequencies)
    const double freq_bin = static_cast<double>(header.sample_rate) / n;
    const size_t useful_bins = n / 2 + 1;

    std::cout << "=== FFT Magnitude (primo segmento, " << n << " punti) ===\n";
    std::cout << "  Bin  |  Freq (Hz)  |  Magnitude\n";
    std::cout << "  -----|-------------|------------\n";

    double max_mag = 0.0;
    size_t max_bin = 0;
    for (size_t i = 0; i < useful_bins; ++i) {
        double mag = std::abs(buf[i]) / n;
        if (mag > max_mag) { max_mag = mag; max_bin = i; }
    }

    // print top bins (magnitudine > 1% del picco) to avoid flooding stdout
    for (size_t i = 0; i < useful_bins; ++i) {
        double mag = std::abs(buf[i]) / n;
        if (mag > max_mag * 0.01) {
            std::cout << "  " << i
                      << "\t|  " << i * freq_bin
                      << "\t|  " << mag << "\n";
        }
    }

    std::cout << "\n=== Picco dominante ===\n"
              << "  Bin " << max_bin
              << " -> " << max_bin * freq_bin << " Hz"
              << " (mag: " << max_mag << ")\n";

    return 0;
}
