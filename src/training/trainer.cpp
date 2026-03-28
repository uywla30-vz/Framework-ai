#include "trainer.h"
#include <cmath>
namespace hwsb {
Trainer::Trainer(LanguageModel& m, Dataset& d) : model(m), dataset(d) {
    m_mu = v_mu = m_sigma = v_sigma = Eigen::VectorXd::Zero(m.get_K());
}
void Trainer::train_step(int bs) {
    std::vector<std::vector<int>> in, tg; if (!dataset.next_batch(bs, in, tg)) { dataset.reset(); dataset.next_batch(bs, in, tg); }
    Eigen::MatrixXd logits = model.forward(in);
    for (int i=0; i<logits.rows(); ++i) { double rm = logits.row(i).maxCoeff(); logits.row(i).array() -= rm; }
    Eigen::MatrixXd p = logits.array().exp().matrix();
    Eigen::VectorXd sums = p.rowwise().sum();
    for (int i=0; i<p.rows(); ++i) p.row(i) /= (sums(i)+1e-10);
    Eigen::MatrixXd dL = p; int tt = (int)in.size() * (int)in[0].size();
    for (int b=0; b<(int)in.size(); ++b) for (int i=0; i<(int)in[0].size(); ++i) dL(b * in[0].size() + i, tg[b][i]) -= 1.0;
    dL /= (double)tt; model.backward(dL); model.update_params(0.001);
}
}
