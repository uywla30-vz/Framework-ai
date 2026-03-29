#include "layer.h"
#include <iostream>
using namespace hwsb;
int main() {
    HWSBLayer l(256, 512, 50);
    std::cout << "HWSBLayer size: " << sizeof(l) << " bytes" << std::endl;
    if (sizeof(l) < 200) { std::cout << "PASSED" << std::endl; return 0; }
    std::cout << "FAILED" << std::endl; return 1;
}
