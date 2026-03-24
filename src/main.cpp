#include "wav.h"
#include "steno.h"
#include "steno_freq.h"
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "StenoWAV v0.2.0 - Scatter-Embed Audio Steganography\n\n"
              << "Usage:\n"
              << "  " << prog << " encode      [--dither] <input.wav> <output.wav> \"message\"\n"
              << "  " << prog << " decode      <input.wav> <cipher>\n"
              << "  " << prog << " encode-freq <input.wav> <output.wav> \"message\"\n"
              << "  " << prog << " decode-freq <input.wav> <cipher>\n\n"
              << "Modes:\n"
              << "  encode/decode            Time-domain LSB scatter-embed\n"
              << "  encode-freq/decode-freq  Frequency-domain QIM scatter-embed (16-bit only)\n\n"
              << "Options:\n"
              << "  --dither  (encode only) Add LSB noise to mask embed positions in FFT analysis\n\n"
              << "The cipher is auto-generated during encoding and required for decoding.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "encode") {
        // Parse optional --dither flag
        bool dither = false;
        int argOffset = 2;
        if (argc > 2 && std::string(argv[2]) == "--dither") {
            dither = true;
            argOffset = 3;
        }

        if (argc - argOffset != 3) {
            std::cerr << "Error: encode requires 3 arguments\n\n";
            print_usage(argv[0]);
            return 1;
        }

        std::string inputPath = argv[argOffset];
        std::string outputPath = argv[argOffset + 1];
        std::string message = argv[argOffset + 2];

        Wav wav(inputPath);

        std::cout << "WAV loaded: " << wav.SampleCount << " samples, "
                  << wav.fmt.BitsPerSample << "-bit, "
                  << wav.fmt.NbrChannel << " channel(s), "
                  << wav.fmt.Frequency << " Hz\n";

        auto result = steno_encode(wav, message, dither);

        if (!result.success) {
            std::cerr << "Error: encoding failed (message too large or not enough valid samples).\n";
            return 1;
        }

        wav.write(outputPath);
        std::cout << "Encoded " << message.size() << " bytes into " << outputPath << "\n";
        std::cout << "Cipher: " << result.cipher << "\n";
        std::cerr << "Save this cipher! You need it to decode.\n";

    } else if (mode == "decode") {
        if (argc != 4) {
            std::cerr << "Error: decode requires 2 arguments (file + cipher)\n\n";
            print_usage(argv[0]);
            return 1;
        }

        std::string inputPath = argv[2];
        std::string cipher = argv[3];

        if (cipher.size() != 16) {
            std::cerr << "Error: cipher must be a 16-character hex string.\n";
            return 1;
        }

        Wav wav(inputPath);

        std::string message = steno_decode(wav, cipher);
        if (message.empty()) {
            std::cerr << "Decoding failed: wrong cipher, corrupted file, or no hidden message.\n";
            return 1;
        }

        std::cout << message << std::endl;

    } else if (mode == "encode-freq") {
        if (argc != 5) {
            std::cerr << "Error: encode-freq requires 3 arguments\n\n";
            print_usage(argv[0]);
            return 1;
        }

        std::string inputPath = argv[2];
        std::string outputPath = argv[3];
        std::string message = argv[4];

        Wav wav(inputPath);

        std::cout << "WAV loaded: " << wav.SampleCount << " samples, "
                  << wav.fmt.BitsPerSample << "-bit, "
                  << wav.fmt.NbrChannel << " channel(s), "
                  << wav.fmt.Frequency << " Hz\n";

        if (wav.fmt.BitsPerSample != 16) {
            std::cerr << "Error: frequency-domain mode requires 16-bit audio.\n";
            return 1;
        }

        auto result = steno_freq_encode(wav, message);

        if (!result.success) {
            std::cerr << "Error: encoding failed (message too large or not enough valid bins).\n";
            return 1;
        }

        wav.write(outputPath);
        std::cout << "Encoded " << message.size() << " bytes (freq-domain) into " << outputPath << "\n";
        std::cout << "Cipher: " << result.cipher << "\n";
        std::cerr << "Save this cipher! You need it to decode.\n";

    } else if (mode == "decode-freq") {
        if (argc != 4) {
            std::cerr << "Error: decode-freq requires 2 arguments (file + cipher)\n\n";
            print_usage(argv[0]);
            return 1;
        }

        std::string inputPath = argv[2];
        std::string cipher = argv[3];

        if (cipher.size() != 16) {
            std::cerr << "Error: cipher must be a 16-character hex string.\n";
            return 1;
        }

        Wav wav(inputPath);
        std::string message = steno_freq_decode(wav, cipher);
        if (message.empty()) {
            std::cerr << "Decoding failed: wrong cipher, corrupted file, or no hidden message.\n";
            return 1;
        }

        std::cout << message << std::endl;

    } else {
        std::cerr << "Error: unknown mode '" << mode << "'\n\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
