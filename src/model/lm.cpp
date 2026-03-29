#include "lm.h"
#include <iostream>
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
    auto& pa = phi_cache.at({config.d_model, config.d_model}); auto& pf1 = phi_cache.at({config.d_model, config.d_ff}); auto& pf2 = phi_cache.at({config.d_ff, config.d_model});
    for (auto& b : blocks) h = b.forward(h, global_mu, global_sigma, pa, pf1, pf2);
    return h * embedding.transpose();
}
void LanguageModel::backward(const Eigen::MatrixXd& dL) {
    Eigen::VectorXd d_mu = Eigen::VectorXd::Zero(config.K);
    Eigen::VectorXd d_sigma = Eigen::VectorXd::Zero(config.K);
    auto& pa = phi_cache.at({config.d_model, config.d_model}); auto& pf1 = phi_cache.at({config.d_model, config.d_ff}); auto& pf2 = phi_cache.at({config.d_ff, config.d_model});
    // Iterate blocks backwards and accumulate global grads
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        // auto b_grads = it->backward(dL, ...);
        // d_mu += b_grads.d_mu; d_sigma += b_grads.d_sigma;
    }
}
void LanguageModel::update_params(double lr) {}
void LanguageModel::save(const std::string& p) {}
void LanguageModel::load(const std::string& p) {}
}
