#include "basis.h"
#include <iostream>
using namespace hwsb;
int main() {
    int K=5, I=4, J=4;
    Eigen::VectorXd a=Eigen::VectorXd::Random(K), b=Eigen::VectorXd::Random(K), g=Eigen::VectorXd::Random(K), m=Eigen::VectorXd::Random(K), s=Eigen::VectorXd::Random(K).array().abs()+0.1;
    Eigen::MatrixXd phi=compute_phi(I, J), dL=Eigen::MatrixXd::Random(I, J);
    BasisGradients ag=compute_gradients(K, dL, b, m, s, phi);
    double eps=1e-6, tol=1e-5;
    auto check=[&](const char* name, Eigen::VectorXd& p, const Eigen::VectorXd& an_g) {
        for (int k=0; k<K; ++k) {
            double v=p(k); p(k)=v+eps; double lp=(synthesize_weights(K,a,b,g,m,s,phi).array()*dL.array()).sum();
            p(k)=v-eps; double lm=(synthesize_weights(K,a,b,g,m,s,phi).array()*dL.array()).sum();
            p(k)=v; double ng=(lp-lm)/(2.0*eps);
            if (std::abs(ng-an_g(k))>tol) { std::cout << name << " failed at " << k << std::endl; return false; }
        }
        return true;
    };
    if (check("a",a,ag.d_alpha) && check("b",b,ag.d_beta) && check("g",g,ag.d_gamma) && check("m",m,ag.d_mu) && check("s",s,ag.d_sigma)) { std::cout << "PASSED" << std::endl; return 0; }
    std::cout << "FAILED" << std::endl; return 1;
}
