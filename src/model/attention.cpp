#include "attention.h"
#include <cmath>
namespace hwsb {
MultiHeadAttention::MultiHeadAttention(int dm, int nh, int K) : d_model(dm), n_heads(nh), d_head(dm/nh), W_q(dm,dm,K), W_k(dm,dm,K), W_v(dm,dm,K), W_out(dm,dm,K) {}
Eigen::MatrixXd MultiHeadAttention::forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po) {
    Eigen::MatrixXd Q = W_q.forward(x, mu, sigma, pq);
    Eigen::MatrixXd K = W_k.forward(x, mu, sigma, pq);
    Eigen::MatrixXd V = W_v.forward(x, mu, sigma, pq);
    Eigen::MatrixXd s = (Q * K.transpose()) / std::sqrt(d_head);
    Eigen::MatrixXd exp_s = (s.colwise() - s.rowwise().maxCoeff()).array().exp().matrix();
    Eigen::VectorXd sums = exp_s.rowwise().sum();
    Eigen::MatrixXd aw = exp_s.array().colwise() / (sums.array() + 1e-10);
    return W_out.forward(aw * V, mu, sigma, po);
}
std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> MultiHeadAttention::backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po) {
    Eigen::MatrixXd Q = W_q.forward(x, mu, sigma, pq);
    Eigen::MatrixXd K = W_k.forward(x, mu, sigma, pq);
    Eigen::MatrixXd V = W_v.forward(x, mu, sigma, pq);
    Eigen::MatrixXd s = (Q * K.transpose()) / std::sqrt(d_head);
    Eigen::MatrixXd exp_s = (s.colwise() - s.rowwise().maxCoeff()).array().exp().matrix();
    Eigen::VectorXd sums = exp_s.rowwise().sum();
    Eigen::MatrixXd aw = exp_s.array().colwise() / (sums.array() + 1e-10);

    auto out_back = W_out.backward(dL_dout, aw * V, mu, sigma, po);
    auto v_back = W_v.backward(aw.transpose() * out_back.first, x, mu, sigma, pq);
    auto q_back = W_q.backward(Eigen::MatrixXd::Zero(x.rows(), d_model), x, mu, sigma, pq);
    auto k_back = W_k.backward(Eigen::MatrixXd::Zero(x.rows(), d_model), x, mu, sigma, pq);
    std::vector<BasisGradients> all_grads = {q_back.second, k_back.second, v_back.second, out_back.second};
    return {q_back.first + k_back.first + v_back.first, all_grads};
}
void MultiHeadAttention::update_params(const std::vector<BasisGradients>& gs, double lr) {
    W_q.update_params(gs[0], lr); W_k.update_params(gs[1], lr); W_v.update_params(gs[2], lr); W_out.update_params(gs[3], lr);
}
void MultiHeadAttention::save(std::ostream& os) {
    W_q.save(os); W_k.save(os); W_v.save(os); W_out.save(os);
}
void MultiHeadAttention::load(std::istream& is) {
    W_q.load(is); W_k.load(is); W_v.load(is); W_out.load(is);
}
}
