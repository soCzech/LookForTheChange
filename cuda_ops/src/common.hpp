#include <torch/extension.h>

// C++ interface
#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_GPU(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a GPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CUDA_INPUT(x) CHECK_GPU(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> optimal_state_change(
    torch::Tensor state_tensor, torch::Tensor action_tensor, torch::Tensor lens, int delta, int kappa, int max_action_state_distance);
