#include <iostream>
#include <string>
#include <vector>
#include "lm.h"
#include "dataset.h"
#include "trainer.h"

using namespace hwsb;

int main() {
    ModelConfig c;
    c.vocab_size = 256; // Matching our simple byte-level binary encoding
    LanguageModel m(c);
    Dataset d("corpus.bin", 64);
    Trainer t(m, d);

    std::cout << "Starting training on corpus.bin..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        t.train_step(4); // batch_size=4
        if (i % 2 == 0) {
            std::cout << "Step " << i << " completed." << std::endl;
        }
    }
    std::cout << "Training finished!" << std::endl;
    m.save("model.bin");
    std::cout << "Model saved to model.bin" << std::endl;

    return 0;
}
