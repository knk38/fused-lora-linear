import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import _cuda as fused_cuda  # built by setup.py
    _HAS_CUDA_EXT = True
except Exception:
    fused_cuda = None
    _HAS_CUDA_EXT = False

class FusedLoRALinear(nn.Module):
    """
    Drop-in linear with LoRA fused at inference:
      y = x W^T + bias + (alpha/r) * (x B^T) A^T
    Shapes:
      W: [out, in], A: [out, r], B: [r, in], x: [B, in]
    """
    def __init__(self, in_features, out_features, r, alpha=1.0, bias=True, impl="auto", dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.impl = impl  # "auto" | "cuda" | "triton" | "naive"

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None

        # single active adapter; extensible to multiple via dict
        self.A = nn.Parameter(torch.empty(out_features, self.r, **factory_kwargs), requires_grad=False)
        self.B = nn.Parameter(torch.empty(self.r, in_features, **factory_kwargs), requires_grad=False)
        self.active = False  # no adapter until initialized

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        # LoRA params start near zero so base W dominates until trained
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)

    def register_adapter(self, name="default", A=None, B=None, alpha=None):
        # Simple single-adapter API; extend to dict if you want multiple.
        if A is not None:
            assert A.shape == (self.out_features, self.r)
            with torch.no_grad(): self.A.copy_(A)
        if B is not None:
            assert B.shape == (self.r, self.in_features)
            with torch.no_grad(): self.B.copy_(B)
        if alpha is not None:
            self.alpha = float(alpha)
        self.active = True

    def set_active_adapter(self, name="default"):
        # Placeholder for a multi-adapter interface
        self.active = True

    @torch.no_grad()
    def merge_weights_(self):
        """Optional: permanently fold A@B into W (disables runtime swapping)."""
        if not self.active: return
        scaling = self.alpha / max(1, self.r)
        self.weight.add_(torch.matmul(self.A, self.B) * scaling)
        # zero out LoRA so forward is pure Linear
        self.A.zero_(); self.B.zero_()
        self.active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback (always correct)
        out = F.linear(x, self.weight, self.bias)
        if not self.active:
            return out

        scaling = self.alpha / max(1, self.r)

        # Choose implementation
        impl = "naive"
        if self.impl in ("auto", "cuda") and _HAS_CUDA_EXT and x.is_cuda and self.weight.is_cuda:
            impl = "cuda"
        elif self.impl == "triton":
            impl = "triton"  # to be implemented

        if impl == "cuda":
            return fused_cuda.lora_linear_forward(x, self.weight, self.A, self.B, self.bias, scaling)
        elif impl == "triton":
            from .kernels.triton.lora_linear_kernel import lora_linear_triton
            return lora_linear_triton(x, self.weight, self.A, self.B, self.bias, scaling)
        else:
            # Naive: (x @ W.T) + (alpha/r) * (x @ B.T) @ A.T
            out = out + scaling * (x @ self.B.t()) @ self.A.t()
            return out
