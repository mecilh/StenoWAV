#ifndef WAV_H
#define WAV_H

#include <fstream>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <variant>
#include <string>

#pragma pack(1)
class Wav {
public:
    class WavHeader {
    public:
        char FileTypeBlocID[4];
        int FileSize;
        char FileFormatID[4];
        char FormatBlockID[4];
        int BlocSize;
        short int AudioFormat;
        short int NbrChannel;
        int Frequency;
        int BytePerSec;
        short int BytePerBloc;
        short int BitsPerSample;
        char DataBloc[4];
        int DataSize;
    };

    class WavData {
    public:
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> RightChannelSamples;
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> LeftChannelSamples;
        std::variant<std::vector<uint8_t>, std::vector<int16_t>> RawChannelSamples;
    };

public:
    WavHeader header;
    WavData data;
    int SampleCount;

    Wav(std::string filePath);

private:
    void read_header(std::ifstream& input);
    int read_data_stereo(std::ifstream& input);
    int read_data_mono(std::ifstream& input);
};
#pragma pack()

#endif