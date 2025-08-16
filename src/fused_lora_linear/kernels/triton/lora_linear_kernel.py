import torch

def lora_linear_triton(x, W, A, B, bias, scaling: float):
    # TODO: write Triton fused kernel. Temporary reference impl:
    out = torch.nn.functional.linear(x, W, bias)
    out = out + (x @ B.t()) @ A.t() * scaling
    return out
