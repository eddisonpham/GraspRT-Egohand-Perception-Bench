"""Stage 01 GPU smoke test — verify sm_120 (Blackwell) works on this torch build."""
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (12, 0)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul OK, result sample:", c[0, 0].item())
    print("peak mem MB:", torch.cuda.max_memory_allocated() / 1e6)
    # FP16 matmul check (needed for FP16 inference + quantization work)
    a16 = a.half()
    b16 = b.half()
    c16 = a16 @ b16
    torch.cuda.synchronize()
    print("fp16 matmul OK, result sample:", c16[0, 0].item())

if __name__ == "__main__":
    pass