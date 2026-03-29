#include "transformer.h"
namespace hwsb {
TransformerBlock::TransformerBlock(int dm, int nh, int dff, int K) : attn(dm, nh, K), ff1(dm, dff, K), ff2(dff, dm, K) {
    g1 = g2 = Eigen::VectorXd::Ones(dm); b1 = b2 = Eigen::VectorXd::Zero(dm);
}
Eigen::MatrixXd layer_norm_impl(const Eigen::MatrixXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& b) {
    Eigen::VectorXd m = x.rowwise().mean();
    Eigen::VectorXd v = (x.rowwise().squaredNorm().array() / x.cols()) - m.array().square();
    Eigen::MatrixXd n = (x.colwise() - m).array().rowwise() / (v.array() + 1e-6).sqrt();
    for(int i=0; i<n.rows(); ++i) n.row(i) = n.row(i).array() * g.transpose().array() + b.transpose().array();
    return n;
}
Eigen::MatrixXd TransformerBlock::forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& si, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2) {
    Eigen::MatrixXd h = x + attn.forward(layer_norm_impl(x, g1, b1), mu, si, pa, pa);
    Eigen::MatrixXd f1 = ff1.forward(layer_norm_impl(h, g2, b2), mu, si, pf1).array().max(0.0);
    return h + ff2.forward(f1, mu, si, pf2);
}
std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> TransformerBlock::backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& si, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2) {
    // Priority 1: Residual backward (dL_dx = dL_dout)
    // Priority 2: Full block backward
    auto f2_back = ff2.backward(dL_dout, Eigen::MatrixXd::Zero(dL_dout.rows(), ff1.get_alpha().size()), mu, si, pf2);
    auto f1_back = ff1.backward(f2_back.first, Eigen::MatrixXd::Zero(dL_dout.rows(), x.cols()), mu, si, pf1);
    auto attn_back = attn.backward(dL_dout, x, mu, si, pa, pa);
    std::vector<BasisGradients> all = attn_back.second;
    all.push_back(f1_back.second); all.push_back(f2_back.second);
    return {dL_dout + f1_back.first + attn_back.first, all};
}
}
