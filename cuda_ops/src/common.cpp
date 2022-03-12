#include "common.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("optimal_state_change", &optimal_state_change, "Optimal State1-Action-State2 Sequence (GPU)");
}
