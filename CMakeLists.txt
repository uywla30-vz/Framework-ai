cmake_minimum_required(VERSION 3.10)
project(hwsb_lm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(third_party)
include_directories(src/model)
include_directories(src/data)
include_directories(src/training)

add_executable(hwsb_lm
    src/main.cpp
    src/model/basis.cpp
    src/model/layer.cpp
    src/model/attention.cpp
    src/model/transformer.cpp
    src/model/lm.cpp
    src/data/tokenizer.cpp
    src/data/dataset.cpp
    src/training/adam.cpp
    src/training/trainer.cpp
)
