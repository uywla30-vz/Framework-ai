#include "basis.h"
namespace hwsb {
Eigen::MatrixXd compute_phi(int I_max, int J_max) {
    Eigen::MatrixXd phi(I_max, J_max); double pi = std::acos(-1.0);
    for (int i = 0; i < I_max; ++i) for (int j = 0; j < J_max; ++j) {
        if (I_max == 1) phi(0, j) = std::cos((j + 0.5) * (2.0 * pi / J_max));
        else phi(i, j) = std::sin((i + 0.5) * (pi / I_max)) * std::cos((j + 0.5) * (2.0 * pi / J_max));
    }
    return phi;
}
double rbf(double phi, double mu, double sigma) { return std::exp(-std::pow(phi - mu, 2) / (2.0 * std::pow(sigma, 2))); }
Eigen::MatrixXd synthesize_weights(int K, const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta, const Eigen::VectorXd& gamma, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi) {
    int I = phi.rows(), J = phi.cols(); Eigen::MatrixXd W = Eigen::MatrixXd::Zero(I, J);
    for (int i = 0; i < I; ++i) for (int j = 0; j < J; ++j) {
        double p = phi(i, j), w = 0.0;
        for (int k = 0; k < K; ++k) {
            double c = std::cos((k+1) * p);
            w += alpha(k) * c + beta(k) * rbf(p, mu(k), sigma(k)) + gamma(k) * (c >= 0 ? 1.0 : -1.0);
        }
        W(i, j) = w;
    }
    return W;
}
BasisGradients compute_gradients(int K, const Eigen::MatrixXd& dL_dW, const Eigen::VectorXd& beta, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi) {
    int I = phi.rows(), J = phi.cols(); BasisGradients g(K);
    for (int i = 0; i < I; ++i) for (int j = 0; j < J; ++j) {
        double p = phi(i, j), d = dL_dW(i, j);
        for (int k = 0; k < K; ++k) {
            double c = std::cos((k+1) * p), r = rbf(p, mu(k), sigma(k));
            g.d_alpha(k) += d * c; g.d_beta(k) += d * r; g.d_gamma(k) += d * (c >= 0 ? 1.0 : -1.0);
            g.d_mu(k) += d * beta(k) * r * (p - mu(k)) / std::pow(sigma(k), 2);
            g.d_sigma(k) += d * beta(k) * r * std::pow(p - mu(k), 2) / std::pow(sigma(k), 3);
        }
    }
    return g;
}
}
