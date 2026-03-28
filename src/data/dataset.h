#ifndef DATASET_H
#define DATASET_H
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
namespace hwsb {
class Dataset {
public:
    Dataset(const std::string& p, int sl, size_t bsb = 64 * 1024 * 1024);
    bool next_batch(int bs, std::vector<std::vector<int>>& in, std::vector<std::vector<int>>& tg);
    void reset();
private:
    std::string path; int seq_len; size_t buffer_size, buffer_pos, buffer_end;
    std::ifstream file; std::vector<uint16_t> buffer;
    void refill_buffer();
};
}
#endif
