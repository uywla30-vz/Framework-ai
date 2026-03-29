#ifndef LM_H
#define LM_H
#include "layer.h"
#include "transformer.h"
#include <vector>
#include <map>
namespace hwsb {
struct ModelConfig { int vocab_size=8000, d_model=256, n_layers=6, n_heads=4, d_ff=512, K=50, max_seq_len=512; };
class LanguageModel {
public:
    LanguageModel(const ModelConfig& c);
    Eigen::MatrixXd forward(const std::vector<std::vector<int>>& x);
    void backward(const Eigen::MatrixXd& dL);
    void update_params(double lr);
    int get_K() const { return config.K; }
    void save(const std::string& p);
    void load(const std::string& p);
private:
    ModelConfig config; Eigen::VectorXd global_mu, global_sigma; Eigen::MatrixXd embedding;
    std::vector<TransformerBlock> blocks; std::map<std::pair<int, int>, Eigen::MatrixXd> phi_cache;
    std::vector<std::vector<BasisGradients>> last_grads;
    std::vector<Eigen::MatrixXd> block_activations;
    Eigen::MatrixXd last_x_emb;
    void precompute_phis();
};
}
#endif
