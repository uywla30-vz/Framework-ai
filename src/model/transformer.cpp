#include "transformer.h"
namespace hwsb {
TransformerBlock::TransformerBlock(int dm, int nh, int dff, int K) : attn(dm, nh, K), ff1(dm, dff, K), ff2(dff, dm, K) {
    g1 = g2 = Eigen::VectorXd::Ones(dm); b1 = b2 = Eigen::VectorXd::Zero(dm);
}
Eigen::MatrixXd layer_norm_impl(const Eigen::MatrixXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& b) {
    Eigen::VectorXd m = x.rowwise().mean();
    Eigen::VectorXd v = (x.rowwise().squaredNorm().array() / x.cols()) - m.array().square();
    Eigen::MatrixXd n = (x.colwise() - m).array().colwise() / (v.array() + 1e-6).sqrt();
    n = (n.array().rowwise() * g.transpose().array()).rowwise() + b.transpose().array();
    return n;
}
Eigen::MatrixXd TransformerBlock::forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& si, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2) {
    Eigen::MatrixXd h = x + attn.forward(layer_norm_impl(x, g1, b1), mu, si, pa, pa);
    Eigen::MatrixXd f1 = ff1.forward(layer_norm_impl(h, g2, b2), mu, si, pf1).array().max(0.0);
    return h + ff2.forward(f1, mu, si, pf2);
}
std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> TransformerBlock::backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& si, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2) {
    // Note: Re-calculating internal activations for backward
    Eigen::MatrixXd h_ln = layer_norm_impl(x, g1, b1);
    Eigen::MatrixXd attn_out = attn.forward(h_ln, mu, si, pa, pa);
    Eigen::MatrixXd h = x + attn_out;
    Eigen::MatrixXd h2_ln = layer_norm_impl(h, g2, b2);
    Eigen::MatrixXd f1 = ff1.forward(h2_ln, mu, si, pf1).array().max(0.0);

    auto f2_back = ff2.backward(dL_dout, f1, mu, si, pf2);
    Eigen::MatrixXd df1 = f2_back.first;
    for(int i=0; i<f1.rows(); ++i) for(int j=0; j<f1.cols(); ++j) if(f1(i,j) <= 0) df1(i,j) = 0;
    auto f1_back = ff1.backward(df1, h2_ln, mu, si, pf1);
    auto attn_back = attn.backward(dL_dout, h_ln, mu, si, pa, pa);
    std::vector<BasisGradients> all = attn_back.second;
    all.push_back(f1_back.second); all.push_back(f2_back.second);
    return {dL_dout + f1_back.first + attn_back.first, all};
}
void TransformerBlock::update_params(const std::vector<BasisGradients>& gs, double lr) {
    std::vector<BasisGradients> attn_gs(gs.begin(), gs.begin()+4);
    attn.update_params(attn_gs, lr); ff1.update_params(gs[4], lr); ff2.update_params(gs[5], lr);
}
void TransformerBlock::save(std::ostream& os) {
    attn.save(os); ff1.save(os); ff2.save(os);
    os.write((char*)g1.data(), g1.size() * sizeof(double));
    os.write((char*)b1.data(), b1.size() * sizeof(double));
    os.write((char*)g2.data(), g2.size() * sizeof(double));
    os.write((char*)b2.data(), b2.size() * sizeof(double));
}
void TransformerBlock::load(std::istream& is) {
    attn.load(is); ff1.load(is); ff2.load(is);
    is.read((char*)g1.data(), g1.size() * sizeof(double));
    is.read((char*)b1.data(), b1.size() * sizeof(double));
    is.read((char*)g2.data(), g2.size() * sizeof(double));
    is.read((char*)b2.data(), b2.size() * sizeof(double));
}
}
