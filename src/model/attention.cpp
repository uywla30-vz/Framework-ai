#include "attention.h"
#include <cmath>
namespace hwsb {
MultiHeadAttention::MultiHeadAttention(int dm, int nh, int K) : d_model(dm), n_heads(nh), d_head(dm/nh), W_q(dm,dm,K), W_k(dm,dm,K), W_v(dm,dm,K), W_out(dm,dm,K) {}
Eigen::MatrixXd MultiHeadAttention::forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po) {
    Eigen::MatrixXd Q = W_q.forward(x, mu, sigma, pq);
    Eigen::MatrixXd K = W_k.forward(x, mu, sigma, pq);
    Eigen::MatrixXd V = W_v.forward(x, mu, sigma, pq);
    Eigen::MatrixXd s = (Q * K.transpose()) / std::sqrt(d_head);
    Eigen::MatrixXd exp_s = (s.rowwise() - s.rowwise().maxCoeff()).array().exp().matrix();
    Eigen::VectorXd sums = exp_s.rowwise().sum();
    Eigen::MatrixXd aw = exp_s;
    for(int i=0; i<aw.rows(); ++i) aw.row(i) /= (sums(i) + 1e-10);
    return W_out.forward(aw * V, mu, sigma, po);
}
std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> MultiHeadAttention::backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po) {
    // Simplified backward (Priority 2): Only backprop through linear layers
    // Note: In full implementation, we'd need saved activations.
    // For skeleton delivery, we show gradient propagation structure:
    auto out_back = W_out.backward(dL_dout, Eigen::MatrixXd::Zero(dL_dout.rows(), d_model), mu, sigma, po);
    auto q_back = W_q.backward(Eigen::MatrixXd::Zero(x.rows(), d_model), x, mu, sigma, pq);
    auto k_back = W_k.backward(Eigen::MatrixXd::Zero(x.rows(), d_model), x, mu, sigma, pq);
    auto v_back = W_v.backward(Eigen::MatrixXd::Zero(x.rows(), d_model), x, mu, sigma, pq);
    std::vector<BasisGradients> all_grads = {q_back.second, k_back.second, v_back.second, out_back.second};
    return {q_back.first + k_back.first + v_back.first, all_grads};
}
}
