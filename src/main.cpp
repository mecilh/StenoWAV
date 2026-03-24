#include "wav.h"
#include "steno.h"
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "StenoWAV v0.2.0 - Scatter-Embed Audio Steganography\n\n"
              << "Usage:\n"
              << "  " << prog << " encode <input.wav> <output.wav> \"message\"\n"
              << "  " << prog << " decode <input.wav> <cipher>\n\n"
              << "The cipher is auto-generated during encoding and required for decoding.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "encode") {
        if (argc != 5) {
            std::cerr << "Error: encode requires 3 arguments\n\n";
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

        auto result = steno_encode(wav, message);

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

    } else {
        std::cerr << "Error: unknown mode '" << mode << "'\n\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
