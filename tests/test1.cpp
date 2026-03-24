#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <fstream>


using boost::dynamic_bitset;



#pragma pack(1)
class WavHeader{
    public:
    int FileTypeBlocID;
    int FileSize;
    int FileFormatID;

    //

    int FormatBlockID;
    int BlocSize;
    short int AudioFormat;
    short int NbrChannel;
    int Frequency;
    int BytePerSec;
    short int BytePerBloc;
    short int BitsPerSample;

    //
};


dynamic_bitset<> byterize(int& size_data, void* data_ptr);



int main(int argc, const char *argv[]){
    std::ifstream fin(argv[1], std::ios::binary);
    WavHeader header_tested;    
    fin.read((char*)&header_tested, sizeof(header_tested));
    dynamic_bitset<> FileTypeBlocID_bits(sizeof(int) * 8, (unsigned long)header_tested.FileTypeBlocID);
    dynamic_bitset<> FileSize_bits(sizeof(int) * 8, (unsigned long)header_tested.FileSize);
    dynamic_bitset<> FileFormatID_bits(sizeof(int) * 8, (unsigned long)header_tested.FileFormatID);
    dynamic_bitset<> FormatBlockID_bits(sizeof(int) * 8, (unsigned long)header_tested.FormatBlockID);
    dynamic_bitset<> BlocSize_bits(sizeof(int) * 8, (unsigned long)header_tested.BlocSize);
    dynamic_bitset<> AudioFormat_bits(sizeof(short int) * 8, (unsigned long)header_tested.AudioFormat);
    dynamic_bitset<> NbrChannel_bits(sizeof(short int) * 8, (unsigned long)header_tested.NbrChannel);
    dynamic_bitset<> Frequency_bits(sizeof(int) * 8, (unsigned long)header_tested.Frequency);
    dynamic_bitset<> BytePerSec_bits(sizeof(int) * 8, (unsigned long)header_tested.BytePerSec);
    dynamic_bitset<> BytePerBloc_bits(sizeof(short int) * 8, (unsigned long)header_tested.BytePerBloc);
    dynamic_bitset<> BitsPerSample_bits(sizeof(short int) * 8, (unsigned long)header_tested.BitsPerSample);

    std::cout << "FileTypeBlocID:  " << FileTypeBlocID_bits << "  (" << header_tested.FileTypeBlocID << ")\n";
    std::cout << "FileSize:        " << FileSize_bits << "  (" << header_tested.FileSize << ")\n";
    std::cout << "FileFormatID:    " << FileFormatID_bits << "  (" << header_tested.FileFormatID << ")\n";
    std::cout << "FormatBlockID:   " << FormatBlockID_bits << "  (" << header_tested.FormatBlockID << ")\n";
    std::cout << "BlocSize:        " << BlocSize_bits << "  (" << header_tested.BlocSize << ")\n";
    std::cout << "AudioFormat:     " << AudioFormat_bits << "  (" << header_tested.AudioFormat << ")\n";
    std::cout << "NbrChannel:      " << NbrChannel_bits << "  (" << header_tested.NbrChannel << ")\n";
    std::cout << "Frequency:       " << Frequency_bits << "  (" << header_tested.Frequency << ")\n";
    std::cout << "BytePerSec:      " << BytePerSec_bits << "  (" << header_tested.BytePerSec << ")\n";
    std::cout << "BytePerBloc:     " << BytePerBloc_bits << "  (" << header_tested.BytePerBloc << ")\n";
    std::cout << "BitsPerSample:   " << BitsPerSample_bits << "  (" << header_tested.BitsPerSample << ")\n";



    
}