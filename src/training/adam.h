#ifndef ADAM_H
#define ADAM_H
#include <Eigen/Dense>
namespace hwsb {
class Adam {
public:
    Adam(double lr=0.001, double b1=0.9, double b2=0.999, double eps=1e-8);
    void step(Eigen::VectorXd& p, Eigen::VectorXd& m, Eigen::VectorXd& v, const Eigen::VectorXd& g);
    void step(Eigen::MatrixXd& p, Eigen::MatrixXd& m, Eigen::MatrixXd& v, const Eigen::MatrixXd& g);
private:
    double lr, b1, b2, eps; int t = 0;
};
}
#endif
