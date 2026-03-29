#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "lm.h"
#include "tokenizer.h"
using namespace hwsb;
int sample(const Eigen::VectorXd& l, double t, double p) {
    Eigen::VectorXd pr = (l.array()/t).exp(); pr /= pr.sum();
    std::vector<std::pair<double, int>> pi; for (int i=0; i<pr.size(); ++i) pi.push_back({pr(i), i});
    std::sort(pi.rbegin(), pi.rend());
    double cp=0; std::vector<int> ti; for (auto& x : pi) { cp+=x.first; ti.push_back(x.second); if (cp>=p) break; }
    Eigen::VectorXd tp(ti.size()); for (size_t i=0; i<ti.size(); ++i) tp(i)=pr(ti[i]); tp/=tp.sum();
    static std::mt19937 g(42); std::discrete_distribution<> d(tp.data(), tp.data()+tp.size()); return ti[d(g)];
}
int main() {
    ModelConfig c; LanguageModel m(c); Tokenizer t; std::string p;
    std::cout << "HWS-B LM v0.1 Ready" << std::endl;
    while (std::cout << "> " << std::flush && std::getline(std::cin, p) && p != "exit") {
        if(p.empty()) continue;
        std::vector<int> ts = t.encode(p); std::cout << "> ";
        for (int i=0; i<20; ++i) {
            Eigen::MatrixXd out = m.forward({ts});
            int nt = sample(out.row(out.rows()-1), 0.8, 0.9);
            ts.push_back(nt); std::cout << t.decode({nt}) << std::flush;
        }
        std::cout << std::endl;
    }
    return 0;
}
