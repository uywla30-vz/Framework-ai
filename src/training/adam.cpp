#include "adam.h"
#include <cmath>
namespace hwsb {
Adam::Adam(double lr, double b1, double b2, double eps) : lr(lr), b1(b1), b2(b2), eps(eps) {}
void Adam::step(Eigen::VectorXd& p, Eigen::VectorXd& m, Eigen::VectorXd& v, const Eigen::VectorXd& g) {
    t++; m = b1 * m + (1.0 - b1) * g; v = b2 * v + (1.0 - b2) * g.array().square().matrix();
    double mh = 1.0 / (1.0 - std::pow(b1, t)), vh = 1.0 / (1.0 - std::pow(b2, t));
    p.array() -= lr * (m.array() * mh) / ((v.array() * vh).sqrt() + eps);
}
void Adam::step(Eigen::MatrixXd& p, Eigen::MatrixXd& m, Eigen::MatrixXd& v, const Eigen::MatrixXd& g) {
    t++; m = b1 * m + (1.0 - b1) * g; v = b2 * v + (1.0 - b2) * g.array().square().matrix();
    double mh = 1.0 / (1.0 - std::pow(b1, t)), vh = 1.0 / (1.0 - std::pow(b2, t));
    p.array() -= lr * (m.array() * mh) / ((v.array() * vh).sqrt() + eps);
}
}
