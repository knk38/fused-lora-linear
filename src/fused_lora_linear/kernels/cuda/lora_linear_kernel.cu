#include <torch/extension.h>

// TODO: Implement a tiled kernel that accumulates x·Wᵀ and (α/r)·x·Bᵀ·Aᵀ.
// For now, call back into PyTorch as a placeholder to keep imports working.
torch::Tensor lora_linear_forward_cuda(
    torch::Tensor x, torch::Tensor W, torch::Tensor A, torch::Tensor B,
    c10::optional<torch::Tensor> bias, double scaling) {
  auto out = at::linear(x, W, bias);
  auto delta = at::mm(at::mm(x, B.t()), A.t()) * scaling;
  return out + delta;
}
