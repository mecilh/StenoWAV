#include <fstream>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <variant>
#include <iostream>

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

private:
    void read_header(std::ifstream& input) {
        input.read((char*)&this->header, sizeof(WavHeader));
        this->SampleCount = header.DataSize / (header.BitsPerSample / 8);
    }

    int read_data_stereo(std::ifstream& input) {
        int sizeSample = this->header.BitsPerSample;

        if (sizeSample == 8) {
            this->data.RawChannelSamples = std::vector<uint8_t>(SampleCount);
            this->data.RightChannelSamples = std::vector<uint8_t>();
            this->data.LeftChannelSamples = std::vector<uint8_t>();
        } else if (sizeSample == 16) {
            this->data.RawChannelSamples = std::vector<int16_t>(SampleCount);
            this->data.RightChannelSamples = std::vector<int16_t>();
            this->data.LeftChannelSamples = std::vector<int16_t>();
        }

        std::visit([&](auto& vec) {
            input.read((char*)vec.data(), this->header.DataSize);
        }, this->data.RawChannelSamples);

        std::visit([&](auto& rawVec) {
            using T = typename std::decay_t<decltype(rawVec)>::value_type;
            auto& left = std::get<std::vector<T>>(this->data.LeftChannelSamples);
            auto& right = std::get<std::vector<T>>(this->data.RightChannelSamples);
            for (int i = 0; i + 1 < (int)rawVec.size(); i += 2) {
                left.push_back(rawVec[i]);
                right.push_back(rawVec[i + 1]);
            }
        }, this->data.RawChannelSamples);

        return 0;
    }

    int read_data_mono(std::ifstream& input) {
        int sizeSample = this->header.BitsPerSample;

        if (sizeSample == 8) {
            this->data.RawChannelSamples = std::vector<uint8_t>(SampleCount);
        } else if (sizeSample == 16) {
            this->data.RawChannelSamples = std::vector<int16_t>(SampleCount);
        }

        std::visit([&](auto& vec) {
            input.read((char*)vec.data(), this->header.DataSize);
        }, this->data.RawChannelSamples);

        return 0;
    }

public:
    Wav(std::string filePath) {
        std::ifstream fin(filePath, std::ios::binary);
        if (!fin.is_open()) {
            std::cerr << "Error opening Wav file" << std::endl;
            exit(1);
        }

        this->read_header(fin);
        int n_channel = this->header.NbrChannel;
        int ret = 1;

        if (n_channel == 1) {
            ret = this->read_data_mono(fin);
        } else if (n_channel == 2) {
            ret = this->read_data_stereo(fin);
        }

        if (ret) {
            std::cerr << "Error reading Wav data" << std::endl;
            exit(1);
        }
    }
};
#pragma pack()