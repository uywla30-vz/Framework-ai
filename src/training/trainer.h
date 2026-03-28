#ifndef TRAINER_H
#define TRAINER_H
#include "lm.h"
#include "dataset.h"
#include "adam.h"
namespace hwsb {
class Trainer {
public:
    Trainer(LanguageModel& m, Dataset& d);
    void train_step(int bs);
private:
    LanguageModel& model; Dataset& dataset; Adam optimizer;
    Eigen::VectorXd m_mu, v_mu, m_sigma, v_sigma;
};
}
#endif
