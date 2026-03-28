#include "dataset.h"
#include <cstdint>
namespace hwsb {
Dataset::Dataset(const std::string& p, int sl, size_t bsb) : path(p), seq_len(sl), buffer_size(bsb/2), buffer_pos(0), buffer_end(0) {
    file.open(path, std::ios::binary); buffer.resize(buffer_size); refill_buffer();
}
void Dataset::refill_buffer() {
    if (!file.is_open()) return;
    file.read((char*)buffer.data(), buffer_size * 2); buffer_end = file.gcount() / 2; buffer_pos = 0;
}
void Dataset::reset() { file.clear(); file.seekg(0); refill_buffer(); }
bool Dataset::next_batch(int bs, std::vector<std::vector<int>>& in, std::vector<std::vector<int>>& tg) {
    in.assign(bs, std::vector<int>(seq_len)); tg.assign(bs, std::vector<int>(seq_len));
    for (int b=0; b<bs; ++b) {
        if (buffer_pos + seq_len + 1 > buffer_end) { refill_buffer(); if (buffer_end < (size_t)seq_len + 1) return false; }
        for (int i=0; i<seq_len; ++i) { in[b][i] = buffer[buffer_pos+i]; tg[b][i] = buffer[buffer_pos+i+1]; }
        buffer_pos += seq_len;
    }
    return true;
}
}
