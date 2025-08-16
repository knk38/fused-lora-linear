#include <torch/extension.h>

// Forward decl (implement in .cu)
torch::Tensor lora_linear_forward_cuda(
    torch::Tensor x, torch::Tensor W, torch::Tensor A, torch::Tensor B,
    c10::optional<torch::Tensor> bias, double scaling);

torch::Tensor lora_linear_forward(
    torch::Tensor x, torch::Tensor W, torch::Tensor A, torch::Tensor B,
    c10::optional<torch::Tensor> bias, double scaling) {
  // type/device checks could be added here
  return lora_linear_forward_cuda(x, W, A, B, bias, scaling);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lora_linear_forward", &lora_linear_forward, "Fused LoRA Linear forward (CUDA)");
}
