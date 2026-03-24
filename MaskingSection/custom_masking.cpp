#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <bitset>

using boost::dynamic_bitset;


class DataSection{
    public:
        bool isFreq;
        dynamic_bitset<> mask();
        dynamic_bitset<> data();
        int 

};