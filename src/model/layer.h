#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <vector>
#include "basis.h"
namespace hwsb {
class HWSBLayer {
public:
    HWSBLayer(int I, int J, int K);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi);
    std::pair<Eigen::MatrixXd, BasisGradients> backward(const Eigen::MatrixXd& dL_da, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& phi);
    Eigen::VectorXd& get_alpha() { return alpha; }
    Eigen::VectorXd& get_beta() { return beta; }
    Eigen::VectorXd& get_gamma() { return gamma; }
    Eigen::VectorXd& get_bias() { return bias; }
    void update_params(const BasisGradients& g, double lr);
    void save(std::ostream& os);
    void load(std::istream& is);
private:
    int I, J, K;
    Eigen::VectorXd alpha, beta, gamma, bias;
    Eigen::VectorXd m_alpha, v_alpha, m_beta, v_beta, m_gamma, v_gamma, m_bias, v_bias;
    int t = 0;
};
}
#endif
