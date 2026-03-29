#ifndef ATTENTION_H
#define ATTENTION_H
#include "layer.h"
namespace hwsb {
class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int n_heads, int K);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po);
    std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pq, const Eigen::MatrixXd& po);
    void update_params(const std::vector<BasisGradients>& gs, double lr);
    void save(std::ostream& os);
    void load(std::istream& is);
private:
    int d_model, n_heads, d_head;
    HWSBLayer W_q, W_k, W_v, W_out;
};
}
#endif
