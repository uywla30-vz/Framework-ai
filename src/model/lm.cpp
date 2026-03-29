#include "lm.h"
#include <iostream>
#include <fstream>
namespace hwsb {
LanguageModel::LanguageModel(const ModelConfig& c) : config(c) {
    global_mu = Eigen::VectorXd::Random(c.K); global_sigma = Eigen::VectorXd::Random(c.K).array().abs()+0.1;
    embedding = Eigen::MatrixXd::Random(c.vocab_size, c.d_model) * 0.01; precompute_phis();
    for (int l=0; l<c.n_layers; ++l) blocks.emplace_back(c.d_model, c.n_heads, c.d_ff, c.K);
}
void LanguageModel::precompute_phis() {
    phi_cache[{config.d_model, config.d_model}] = compute_phi(config.d_model, config.d_model);
    phi_cache[{config.d_model, config.d_ff}] = compute_phi(config.d_model, config.d_ff);
    phi_cache[{config.d_ff, config.d_model}] = compute_phi(config.d_ff, config.d_model);
}
Eigen::MatrixXd LanguageModel::forward(const std::vector<std::vector<int>>& x) {
    int bs = (int)x.size(), sl = (int)x[0].size(); Eigen::MatrixXd h(bs*sl, config.d_model);
    for (int b=0; b<bs; ++b) for (int i=0; i<sl; ++i) h.row(b*sl+i) = embedding.row(x[b][i]);
    last_x_emb = h; block_activations.clear();
    auto& pa = phi_cache.at({config.d_model, config.d_model}); auto& pf1 = phi_cache.at({config.d_model, config.d_ff}); auto& pf2 = phi_cache.at({config.d_ff, config.d_model});
    for (auto& b : blocks) { block_activations.push_back(h); h = b.forward(h, global_mu, global_sigma, pa, pf1, pf2); }
    block_activations.push_back(h); // Save the final hidden state
    return h * embedding.transpose();
}
void LanguageModel::backward(const Eigen::MatrixXd& dL) {
    auto& pa = phi_cache.at({config.d_model, config.d_model}); auto& pf1 = phi_cache.at({config.d_model, config.d_ff}); auto& pf2 = phi_cache.at({config.d_ff, config.d_model});
    Eigen::MatrixXd dh = dL * embedding;
    last_grads.clear();
    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
        auto b_back = blocks[i].backward(dh, block_activations[i], global_mu, global_sigma, pa, pf1, pf2);
        dh = b_back.first; last_grads.push_back(b_back.second);
    }
    std::reverse(last_grads.begin(), last_grads.end());
    // Update embedding gradients
    embedding.array() -= 0.001 * (dL.transpose() * block_activations.back()).array();
}
void LanguageModel::update_params(double lr) {
    if (last_grads.empty()) return;
    for (size_t i=0; i<blocks.size(); ++i) blocks[i].update_params(last_grads[i], lr);
}
void LanguageModel::save(const std::string& p) {
    std::ofstream os(p, std::ios::binary); if (!os) return;
    os.write((char*)global_mu.data(), config.K * sizeof(double));
    os.write((char*)global_sigma.data(), config.K * sizeof(double));
    os.write((char*)embedding.data(), config.vocab_size * config.d_model * sizeof(double));
    for (auto& b : blocks) b.save(os);
}
void LanguageModel::load(const std::string& p) {
    std::ifstream is(p, std::ios::binary); if (!is) return;
    is.read((char*)global_mu.data(), config.K * sizeof(double));
    is.read((char*)global_sigma.data(), config.K * sizeof(double));
    is.read((char*)embedding.data(), config.vocab_size * config.d_model * sizeof(double));
    for (auto& b : blocks) b.load(is);
}
}
