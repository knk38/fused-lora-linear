# Fused LoRA Linear (PyTorch + CUDA)

**One-kernel LoRA inference for linear layers**: compute `y = x·Wᵀ + (α/r)·x·Bᵀ·Aᵀ` in a single pass to cut memory traffic and kernel launches. Drop‑in `nn.Linear` replacement with a fallback path, optional Triton kernel, and Hugging Face PEFT patch script.

> Status: MVP. CPU/naive path ready; CUDA/Triton kernel stubs included. PRs welcome.

## Why fuse LoRA?
- **Fewer launches**: avoid separate GEMMs for `x·Wᵀ` and `x·Bᵀ·Aᵀ`.
- **Less memory traffic**: no intermediate `(x·Bᵀ)` allocation.
- **Adapter‑heavy setups**: bigger wins when swapping adapters at runtime (no pre-merge needed).

## Features
- `FusedLoRALinear(in_features, out_features, r, alpha=1.0, bias=True, impl="auto")`
- Registers multiple adapters and switches them at runtime.
- CUDA/Triton stubs to implement a fused kernel; CPU/PyTorch fallback works now.
- Bench + correctness tests.
- Example script to patch a Hugging Face model after applying PEFT LoRA.

## Install

Requires Python 3.9+, PyTorch 2.x, CUDA (optional for GPU path).

```bash
git clone https://github.com/<you>/fused-lora-linear
cd fused-lora-linear
pip install -e .
# (Optional) build CUDA extension
python setup.py develop
