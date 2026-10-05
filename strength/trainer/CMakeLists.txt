file(GLOB SRCS *.cpp)

find_package(pybind11 REQUIRED)
pybind11_add_module(strength_py ${SRCS})
target_link_libraries(
    strength_py
    PUBLIC
    config
    environment
    learner
    utils
    misc
)

# Training module without the TorchScript scorer: it must be importable after PyTorch (ABI-incompatible libtorch).
pybind11_add_module(strength_train_py ${SRCS})
target_compile_definitions(strength_train_py PRIVATE STRENGTH_PY_TRAINING_ONLY STRENGTH_PY_MODULE_NAME=strength_train_py)
target_link_libraries(
    strength_train_py
    PUBLIC
    config
    environment
    learner
    utils
    misc
)