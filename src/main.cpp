#include "wav.h"
#include "steno.h"
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "StenoWAV v0.1.0 - Audio Steganography Tool\n\n"
              << "Usage:\n"
              << "  " << prog << " encode <input.wav> <output.wav> \"message\"\n"
              << "  " << prog << " decode <input.wav>\n";
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

        if (!steno_encode(wav, message)) {
            std::cerr << "Error: message too large for this audio file.\n"
                      << "  Message needs " << (32 + message.size() * 8) << " samples, "
                      << "file has " << wav.SampleCount << " samples.\n";
            return 1;
        }

        wav.write(outputPath);
        std::cout << "Encoded " << message.size() << " bytes into " << outputPath << "\n";

    } else if (mode == "decode") {
        if (argc != 3) {
            std::cerr << "Error: decode requires 1 argument\n\n";
            print_usage(argv[0]);
            return 1;
        }

        std::string inputPath = argv[2];
        Wav wav(inputPath);

        std::string message = steno_decode(wav);
        if (message.empty()) {
            std::cerr << "No hidden message found (or message was empty).\n";
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
