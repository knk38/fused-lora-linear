import torch
from fused_lora_linear import FusedLoRALinear

def test_close_cpu():
    torch.manual_seed(0)
    B, din, dout, r = 2, 32, 48, 4
    layer = FusedLoRALinear(din, dout, r, alpha=8, impl="naive", dtype=torch.float32, device="cpu")
    x = torch.randn(B, din)
    W = torch.randn(dout, din)
    b = torch.randn(dout)
    A = torch.randn(dout, r) * 1e-3
    Bm = torch.randn(r, din) * 1e-3
    with torch.no_grad():
        layer.weight.copy_(W); layer.bias.copy_(b)
        layer.register_adapter(A=A, B=Bm, alpha=8)

    y_ref = torch.nn.functional.linear(x, W, b) + (x @ Bm.t()) @ A.t() * (8/r)
    y = layer(x)
    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
