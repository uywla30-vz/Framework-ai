#ifndef BASIS_H
#define BASIS_H
#include <Eigen/Dense>
#include <cmath>
namespace hwsb {
Eigen::MatrixXd compute_phi(int I_max, int J_max);
double rbf(double phi, double mu, double sigma);
Eigen::MatrixXd synthesize_weights(int K, const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta, const Eigen::VectorXd& gamma, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi);
struct BasisGradients {
    Eigen::VectorXd d_alpha, d_beta, d_gamma, d_mu, d_sigma, d_bias;
    BasisGradients(int K) { d_alpha = d_beta = d_gamma = d_mu = d_sigma = Eigen::VectorXd::Zero(K); }
};
BasisGradients compute_gradients(int K, const Eigen::MatrixXd& dL_dW, const Eigen::VectorXd& beta, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi);
}
#endif
