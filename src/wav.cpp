#include "wav.h"
#include <iostream>
#include <cstring>

Wav::Wav(const std::string& filePath) {
    std::ifstream fin(filePath, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error opening WAV file: " << filePath << std::endl;
        exit(1);
    }

    read_chunks(fin);

    int ret = 1;
    if (fmt.NbrChannel == 1) {
        ret = read_data_mono(fin);
    } else if (fmt.NbrChannel == 2) {
        ret = read_data_stereo(fin);
    }

    if (ret) {
        std::cerr << "Error reading WAV data" << std::endl;
        exit(1);
    }
}

void Wav::read_chunks(std::ifstream& input) {
    // Read RIFF header
    input.read(reinterpret_cast<char*>(&riff), sizeof(RiffHeader));
    if (std::memcmp(riff.ChunkID, "RIFF", 4) != 0 ||
        std::memcmp(riff.Format, "WAVE", 4) != 0) {
        std::cerr << "Not a valid WAV file" << std::endl;
        exit(1);
    }

    // Scan chunks to find "fmt " and "data"
    bool foundFmt = false;
    bool foundData = false;
    char chunkID[4];
    uint32_t chunkSize;

    while (input.read(chunkID, 4) && input.read(reinterpret_cast<char*>(&chunkSize), 4)) {
        if (std::memcmp(chunkID, "fmt ", 4) == 0) {
            input.read(reinterpret_cast<char*>(&fmt), sizeof(FmtChunk));
            // Skip any extra fmt bytes (e.g. extended format)
            if (chunkSize > sizeof(FmtChunk)) {
                input.seekg(chunkSize - sizeof(FmtChunk), std::ios::cur);
            }
            foundFmt = true;
        } else if (std::memcmp(chunkID, "data", 4) == 0) {
            DataSize = chunkSize;
            SampleCount = DataSize / (fmt.BitsPerSample / 8);
            foundData = true;
            break; // stream is now positioned at audio data
        } else {
            // Skip unknown chunk
            input.seekg(chunkSize, std::ios::cur);
        }
    }

    if (!foundFmt || !foundData) {
        std::cerr << "WAV file missing fmt or data chunk" << std::endl;
        exit(1);
    }
}

int Wav::read_data_stereo(std::ifstream& input) {
    int sizeSample = fmt.BitsPerSample;

    if (sizeSample == 8) {
        data.RawChannelSamples = std::vector<uint8_t>(SampleCount);
        data.RightChannelSamples = std::vector<uint8_t>();
        data.LeftChannelSamples = std::vector<uint8_t>();
    } else if (sizeSample == 16) {
        data.RawChannelSamples = std::vector<int16_t>(SampleCount);
        data.RightChannelSamples = std::vector<int16_t>();
        data.LeftChannelSamples = std::vector<int16_t>();
    } else {
        return 1;
    }

    std::visit([&](auto& vec) {
        input.read(reinterpret_cast<char*>(vec.data()), DataSize);
    }, data.RawChannelSamples);

    std::visit([&](auto& rawVec) {
        using T = typename std::decay_t<decltype(rawVec)>::value_type;
        auto& left = std::get<std::vector<T>>(data.LeftChannelSamples);
        auto& right = std::get<std::vector<T>>(data.RightChannelSamples);
        for (int i = 0; i + 1 < static_cast<int>(rawVec.size()); i += 2) {
            left.push_back(rawVec[i]);
            right.push_back(rawVec[i + 1]);
        }
    }, data.RawChannelSamples);

    return 0;
}

int Wav::read_data_mono(std::ifstream& input) {
    int sizeSample = fmt.BitsPerSample;

    if (sizeSample == 8) {
        data.RawChannelSamples = std::vector<uint8_t>(SampleCount);
    } else if (sizeSample == 16) {
        data.RawChannelSamples = std::vector<int16_t>(SampleCount);
    } else {
        return 1;
    }

    std::visit([&](auto& vec) {
        input.read(reinterpret_cast<char*>(vec.data()), DataSize);
    }, data.RawChannelSamples);

    return 0;
}

void Wav::write(const std::string& outputPath) {
    std::ofstream fout(outputPath, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Error opening output file: " << outputPath << std::endl;
        exit(1);
    }

    // Write RIFF header
    fout.write(reinterpret_cast<const char*>(&riff), sizeof(RiffHeader));

    // Write fmt chunk
    char fmtID[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = sizeof(FmtChunk);
    fout.write(fmtID, 4);
    fout.write(reinterpret_cast<const char*>(&fmtSize), 4);
    fout.write(reinterpret_cast<const char*>(&fmt), sizeof(FmtChunk));

    // Write data chunk
    char dataID[4] = {'d', 'a', 't', 'a'};
    fout.write(dataID, 4);
    fout.write(reinterpret_cast<const char*>(&DataSize), 4);

    std::visit([&](auto& vec) {
        fout.write(reinterpret_cast<const char*>(vec.data()), DataSize);
    }, data.RawChannelSamples);
}
