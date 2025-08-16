import argparse, time, torch
import torch.nn.functional as F
from fused_lora_linear import FusedLoRALinear

def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) * 1000.0 / iters

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--in", dest="din", type=int, default=4096)
    p.add_argument("--out", type=int, default=4096)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    x = torch.randn(args.batch, args.din, device=device, dtype=dtype)
    base = torch.nn.Linear(args.din, args.out, bias=True, device=device, dtype=dtype)
    A = torch.zeros(args.out, args.rank, device=device, dtype=dtype)
    B = torch.zeros(args.rank, args.din, device=device, dtype=dtype)
    # small nonzero to avoid zero work
    A.normal_(mean=0, std=1e-3); B.normal_(mean=0, std=1e-3)

    fused = FusedLoRALinear(args.din, args.out, args.rank, alpha=16, impl="auto", dtype=dtype, device=device)
    with torch.no_grad():
        fused.weight.copy_(base.weight)
        if base.bias is not None: fused.bias.copy_(base.bias)
        fused.register_adapter(A=A, B=B, alpha=16)

    def run_baseline():
        return F.linear(x, base.weight, base.bias) + (x @ B.t()) @ A.t() * (16 / args.rank)

    def run_fused():
        return fused(x)

    t_base = timeit(run_baseline)
    t_fused = timeit(run_fused)
    print(f"baseline (two-GEMM): {t_base:.3f} ms  |  fused: {t_fused:.3f} ms")

if __name__ == "__main__":
    main()
