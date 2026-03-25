"""
GPU VRAM 기반 최적 batch size 자동 탐색.

원리:
  1. batch 1, 2, 4, 8, 16, 32, 64 순서로 더미 forward pass 실행
  2. 각 batch에서 실제 VRAM 사용량 측정
  3. (batch_size, memory) 관계를 1차 선형회귀로 피팅
  4. 목표 VRAM 사용률(기본 60%)에 해당하는 batch size 역산

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/autobatch.py
  CUDA_VISIBLE_DEVICES=2 python scripts/autobatch.py --imgsz 640 --fraction 0.7
"""

import argparse
import torch
import numpy as np


def get_gpu_memory(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1 << 30)
    reserved = torch.cuda.memory_reserved(device) / (1 << 30)
    allocated = torch.cuda.memory_allocated(device) / (1 << 30)
    free = total - (reserved + allocated)
    return {"name": props.name, "total": total, "reserved": reserved, "allocated": allocated, "free": free}


def autobatch(model: torch.nn.Module, imgsz: int = 640, fraction: float = 0.6) -> int:
    device = next(model.parameters()).device
    mem = get_gpu_memory(device)

    print(f"\nGPU: {mem['name']}")
    print(f"  Total:     {mem['total']:.2f} GB")
    print(f"  Reserved:  {mem['reserved']:.2f} GB")
    print(f"  Allocated: {mem['allocated']:.2f} GB")
    print(f"  Free:      {mem['free']:.2f} GB")
    print(f"  Target:    {fraction*100:.0f}% = {mem['total']*fraction:.2f} GB\n")

    batch_sizes = [1, 2, 4, 8, 16] if mem["total"] < 16 else [1, 2, 4, 8, 16, 32, 64]
    results = []

    print(f"{'Batch':>8}  {'VRAM (GB)':>10}  {'Status':>8}")
    print("-" * 32)

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            dummy = torch.zeros(bs, 3, imgsz, imgsz, device=device)
            with torch.no_grad():
                model(dummy)

            vram_used = torch.cuda.max_memory_allocated(device) / (1 << 30)
            results.append((bs, vram_used))
            print(f"{bs:>8}  {vram_used:>10.2f}  {'OK':>8}")

            del dummy
            torch.cuda.empty_cache()

            if vram_used > mem["total"] * fraction:
                break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{bs:>8}  {'OOM':>10}  {'FAIL':>8}")
                torch.cuda.empty_cache()
                break
            raise

    if len(results) < 2:
        print("\nNot enough data points. Using batch=1.")
        return 1

    x = [r[0] for r in results]
    y = [r[1] for r in results]

    p = np.polyfit(x, y, deg=1)
    target_mem = mem["free"] * fraction
    optimal = int((target_mem - p[1]) / p[0])
    optimal = max(1, min(optimal, 1024))

    predicted_mem = np.polyval(p, optimal) + mem["reserved"] + mem["allocated"]
    predicted_frac = predicted_mem / mem["total"]

    print(f"\n{'='*32}")
    print(f"Linear fit: VRAM = {p[0]:.4f} * batch + {p[1]:.4f}")
    print(f"Optimal batch size: {optimal}")
    print(f"Predicted VRAM: {predicted_mem:.2f} GB / {mem['total']:.2f} GB ({predicted_frac*100:.0f}%)")

    return optimal


def main():
    parser = argparse.ArgumentParser(description="Auto batch size finder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--model", type=str, default="yolo26m.pt")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model).model.to("cuda:0")
    model.eval()

    optimal = autobatch(model, imgsz=args.imgsz, fraction=args.fraction)
    print(f"\n>>> Recommended: batch={optimal}")


if __name__ == "__main__":
    main()
