#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <string>
#include <vector>
#include <map>
#include <cstdint>
namespace hwsb {
class Tokenizer {
public:
    Tokenizer();
    void train(const std::string& cp, int tvs);
    std::vector<int> encode(const std::string& t) const;
    std::string decode(const std::vector<int>& ts) const;
    void save(const std::string& vp, const std::string& mp) const;
    void load(const std::string& vp, const std::string& mp);
    int get_vocab_size() const { return (int)id_to_token.size(); }
    static constexpr int BOS_TOKEN = 8001, EOS_TOKEN = 8002, PAD_TOKEN = 8003;
private:
    std::map<int, std::vector<uint8_t>> id_to_token; std::map<std::vector<uint8_t>, int> token_to_id; std::map<std::pair<int, int>, int> merges;
    void initialize_base_vocab();
};
}
#endif
