"""
Step 0.1 — GPU Validation Script
Verifies torch sees CUDA, prints device info and VRAM.
"""

import sys
import torch

def main():
    print("=" * 60)
    print("GPU & Environment Validation")
    print("=" * 60)

    # Python version
    print(f"\nPython version : {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\ntorch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA is NOT available. Check your PyTorch installation.")
        print("   Installed with: --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    # Device info
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_mb = props.total_memory / (1024 ** 2)
        free_vram_mb = torch.cuda.mem_get_info(i)[0] / (1024 ** 2)
        used_vram_mb = total_vram_mb - free_vram_mb

        print(f"\n--- GPU {i}: {props.name} ---")
        print(f"  Compute capability : {props.major}.{props.minor}")
        print(f"  Total VRAM         : {total_vram_mb:,.0f} MB ({total_vram_mb / 1024:.1f} GB)")
        print(f"  Free VRAM          : {free_vram_mb:,.0f} MB ({free_vram_mb / 1024:.1f} GB)")
        print(f"  Used VRAM          : {used_vram_mb:,.0f} MB")
        print(f"  Multi-processor cnt: {props.multi_processor_count}")

    # CUDA version info
    print(f"\nCUDA version (runtime): {torch.version.cuda}")
    print(f"cuDNN version        : {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled        : {torch.backends.cudnn.enabled}")

    # Quick sanity check — allocate a tensor on GPU
    print("\n--- Quick GPU tensor test ---")
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x.T)
    print(f"  Created 1000x1000 tensor on GPU, matmul result shape: {y.shape}")
    print(f"  Result dtype: {y.dtype}, device: {y.device}")
    del x, y
    torch.cuda.empty_cache()

    print("\n✅ GPU validation passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
