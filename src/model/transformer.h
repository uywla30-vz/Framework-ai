#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include "attention.h"
#include "layer.h"
namespace hwsb {
class TransformerBlock {
public:
    TransformerBlock(int d_model, int n_heads, int d_ff, int K);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2);
    std::pair<Eigen::MatrixXd, std::vector<BasisGradients>> backward(const Eigen::MatrixXd& dL_dout, const Eigen::MatrixXd& x, const Eigen::VectorXd& mu, const Eigen::VectorXd& sigma, const Eigen::MatrixXd& pa, const Eigen::MatrixXd& pf1, const Eigen::MatrixXd& pf2);
    void update_params(const std::vector<BasisGradients>& gs, double lr);
    void save(std::ostream& os);
    void load(std::istream& is);
private:
    MultiHeadAttention attn; HWSBLayer ff1, ff2;
    Eigen::VectorXd g1, b1, g2, b2;
};
}
#endif
