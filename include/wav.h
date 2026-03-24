#ifndef WAV_H
#define WAV_H

#include <fstream>
#include <vector>
#include <variant>
#include <cstdint>
#include <string>

class Wav {
public:
#pragma pack(push, 1)
    struct RiffHeader {
        char ChunkID[4];       // "RIFF"
        uint32_t ChunkSize;
        char Format[4];        // "WAVE"
    };

    struct FmtChunk {
        uint16_t AudioFormat;
        uint16_t NbrChannel;
        uint32_t Frequency;
        uint32_t BytePerSec;
        uint16_t BytePerBloc;
        uint16_t BitsPerSample;
    };
#pragma pack(pop)

    struct WavData {
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> RightChannelSamples;
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> LeftChannelSamples;
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> RawChannelSamples;
    };

    RiffHeader riff;
    FmtChunk fmt;
    uint32_t DataSize;
    WavData data;
    int SampleCount;

    Wav(const std::string& filePath);
    void write(const std::string& outputPath);

private:
    void read_chunks(std::ifstream& input);
    int read_data_stereo(std::ifstream& input);
    int read_data_mono(std::ifstream& input);
};

#endif
