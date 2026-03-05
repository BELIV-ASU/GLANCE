#!/usr/bin/env python3
"""
compute_depth.py – Pre-compute depth maps for all nuscenes camera images
using Depth Anything V2 Small.

Saves depth maps as 16-bit grayscale PNGs alongside the originals in a
parallel directory tree:
  /scratch/rbaskar5/Dataset/nuscenes/depth/CAM_FRONT/<same_filename>.png

Usage:
  python3 compute_depth.py
  python3 compute_depth.py --batch_size 8 --num_workers 4
  python3 compute_depth.py --camera CAM_FRONT   # single camera only
"""

import os, sys, argparse, time, glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation


# ---------------------------------------------------------------------------
#  Dataset of nuscenes images
# ---------------------------------------------------------------------------
class NuScenesImageDataset(Dataset):
    def __init__(self, image_paths, depth_root):
        self.image_paths = image_paths
        self.depth_root = depth_root

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        return {"path": path, "idx": idx}


def collect_images(samples_root, depth_root, cameras=None):
    """Collect all images that don't already have a depth map."""
    if cameras is None:
        cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    todo = []
    skipped = 0
    for cam in cameras:
        cam_dir = os.path.join(samples_root, cam)
        if not os.path.isdir(cam_dir):
            print(f"  [!] Camera dir not found: {cam_dir}")
            continue
        for fname in sorted(os.listdir(cam_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(cam_dir, fname)
            # Output: depth/<cam>/<filename_without_ext>.png
            stem = Path(fname).stem
            dst = os.path.join(depth_root, cam, f"{stem}.png")
            if os.path.isfile(dst):
                skipped += 1
                continue
            todo.append((src, dst))

    print(f"  Images to process: {len(todo)}")
    if skipped:
        print(f"  Already computed (skipped): {skipped}")
    return todo


def save_depth_16bit(depth_tensor, dst_path, original_size):
    """Save a depth tensor as a 16-bit grayscale PNG at original resolution.

    Args:
        depth_tensor: (H, W) float tensor (relative depth, larger = further)
        dst_path: output file path
        original_size: (width, height) of the source image
    """
    depth_np = depth_tensor.cpu().numpy()

    # Normalize to 0–65535 range
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)

    depth_16 = (depth_norm * 65535).astype(np.uint16)

    # Resize to original resolution
    depth_img = Image.fromarray(depth_16, mode="I;16")
    depth_img = depth_img.resize(original_size, Image.BILINEAR)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    depth_img.save(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute depth maps with Depth Anything V2")
    parser.add_argument("--samples_root", default="/scratch/rbaskar5/Dataset/nuscenes/samples",
                        help="Root of nuscenes samples (contains CAM_FRONT/, etc.)")
    parser.add_argument("--depth_root", default="/scratch/rbaskar5/Dataset/nuscenes/depth",
                        help="Output root for depth maps")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--camera", default=None,
                        help="Process only this camera (e.g. CAM_FRONT). Default: all 6.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cameras = [args.camera] if args.camera else None

    print(f"[*] Depth model: {args.model}")
    print(f"[*] Samples root: {args.samples_root}")
    print(f"[*] Depth output: {args.depth_root}")

    # Collect work
    pairs = collect_images(args.samples_root, args.depth_root, cameras)
    if not pairs:
        print("[*] Nothing to do – all depth maps already exist.")
        return

    # Load model
    print("[*] Loading Depth Anything V2 Small...")
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    model = DepthAnythingForDepthEstimation.from_pretrained(args.model)
    model = model.to(args.device).eval()
    print(f"[*] Model loaded on {args.device}")

    # Process
    t0 = time.time()
    total = len(pairs)
    done = 0
    errors = 0

    for i in range(0, total, args.batch_size):
        batch_pairs = pairs[i : i + args.batch_size]

        # Load images
        pil_images = []
        valid_pairs = []
        for src, dst in batch_pairs:
            try:
                img = Image.open(src).convert("RGB")
                pil_images.append(img)
                valid_pairs.append((src, dst, img.size))  # (W, H)
            except Exception as e:
                print(f"  [!] Error loading {src}: {e}")
                errors += 1

        if not pil_images:
            continue

        # Run depth estimation
        inputs = image_processor(images=pil_images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.predicted_depth: (B, H_model, W_model)
        depth_batch = outputs.predicted_depth

        # Save each depth map
        for j, (src, dst, orig_size) in enumerate(valid_pairs):
            save_depth_16bit(depth_batch[j], dst, orig_size)

        done += len(valid_pairs)

        # Progress
        if done % 100 < args.batch_size or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}]  {rate:.1f} img/s  ETA: {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n[*] Done! Processed {done} images in {elapsed/60:.1f} minutes")
    if errors:
        print(f"[!] Errors: {errors}")
    print(f"[*] Depth maps saved to: {args.depth_root}")


if __name__ == "__main__":
    main()
