#include "layer.h"
namespace hwsb {
HWSBLayer::HWSBLayer(int I, int J, int K) : I(I), J(J), K(K) {
    alpha = Eigen::VectorXd::Random(K) * 0.01;
    beta = Eigen::VectorXd::Random(K) * 0.01;
    gamma = Eigen::VectorXd::Random(K) * 0.01;
    bias = Eigen::VectorXd::Zero(J);
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
}
