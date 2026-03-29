#include "layer.h"
namespace hwsb {
HWSBLayer::HWSBLayer(int I, int J, int K) : I(I), J(J), K(K) {
    alpha = Eigen::VectorXd::Random(K) * 0.01;
    beta = Eigen::VectorXd::Random(K) * 0.01;
    gamma = Eigen::VectorXd::Random(K) * 0.01;
    bias = Eigen::VectorXd::Zero(J);
    m_alpha = v_alpha = m_beta = v_beta = m_gamma = v_gamma = Eigen::VectorXd::Zero(K);
    m_bias = v_bias = Eigen::VectorXd::Zero(J);
}
Eigen::MatrixXd HWSBLayer::forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi) {
    Eigen::MatrixXd W = synthesize_weights(K, alpha, beta, gamma, mu, sigma, phi);
    return (x * W).rowwise() + bias.transpose();
}
std::pair<Eigen::MatrixXd, BasisGradients> HWSBLayer::backward(const Eigen::MatrixXd& dL_da, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi) {
    Eigen::MatrixXd W = synthesize_weights(K, alpha, beta, gamma, mu, sigma, phi);
    Eigen::MatrixXd dL_dx = dL_da * W.transpose();
    Eigen::MatrixXd dL_dW = x.transpose() * dL_da;
    BasisGradients grads = compute_gradients(K, dL_dW, beta, mu, sigma, phi);
    grads.d_bias = dL_da.colwise().sum();
    return {dL_dx, grads};
}
void HWSBLayer::update_params(const BasisGradients& g, double lr) {
    t++; double b1=0.9, b2=0.999, eps=1e-8;
    auto update = [&](Eigen::VectorXd& p, Eigen::VectorXd& m, Eigen::VectorXd& v, const Eigen::VectorXd& grad) {
        m = b1 * m + (1.0 - b1) * grad; v = b2 * v + (1.0 - b2) * grad.array().square().matrix();
        double mh = 1.0 / (1.0 - std::pow(b1, t)), vh = 1.0 / (1.0 - std::pow(b2, t));
        p.array() -= lr * (m.array() * mh) / ((v.array() * vh).sqrt() + eps);
    };
    update(alpha, m_alpha, v_alpha, g.d_alpha);
    update(beta, m_beta, v_beta, g.d_beta);
    update(gamma, m_gamma, v_gamma, g.d_gamma);
    update(bias, m_bias, v_bias, g.d_bias);
}
void HWSBLayer::save(std::ostream& os) {
    os.write((char*)alpha.data(), K * sizeof(double));
    os.write((char*)beta.data(), K * sizeof(double));
    os.write((char*)gamma.data(), K * sizeof(double));
    os.write((char*)bias.data(), J * sizeof(double));
}
void HWSBLayer::load(std::istream& is) {
    is.read((char*)alpha.data(), K * sizeof(double));
    is.read((char*)beta.data(), K * sizeof(double));
    is.read((char*)gamma.data(), K * sizeof(double));
    is.read((char*)bias.data(), J * sizeof(double));
}
}
