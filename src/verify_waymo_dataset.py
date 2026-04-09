#!/usr/bin/env python3
"""
verify_waymo_dataset.py  –  Quick completeness check for the Waymo cache.

Reports the cache size and blob count for the installed Waymo Open Dataset.
The current workspace snapshot is expected to show the full cached dataset
rather than a small sample subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _human_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def verify_cache(cache_root: Path, expected_files: int, min_size_tb: float) -> int:
    blobs_dir = cache_root / "blobs"
    if not blobs_dir.is_dir():
        print(f"Missing blobs directory: {blobs_dir}")
        return 1

    files = [path for path in blobs_dir.iterdir() if path.is_file()]
    total_size = sum(path.stat().st_size for path in files)
    total_tb = total_size / (1024 ** 4)

    print(f"Cache root : {cache_root}")
    print(f"Blob files : {len(files):,}")
    print(f"Total size : {total_tb:.2f} TB")

    if expected_files > 0 and len(files) < expected_files:
        print(f"Warning    : expected at least {expected_files:,} files")
        return 2

    if total_tb < min_size_tb:
        print(f"Warning    : expected at least {min_size_tb:.2f} TB")
        return 3

    print("Status     : looks like the full Waymo cache is present")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the installed Waymo Open Dataset cache")
    parser.add_argument(
        "--cache_root",
        default="/scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3",
        help="Root of the Hugging Face dataset cache",
    )
    parser.add_argument(
        "--expected_files",
        type=int,
        default=1540,
        help="Expected blob file count for the full cache",
    )
    parser.add_argument(
        "--min_size_tb",
        type=float,
        default=3.0,
        help="Minimum expected cache size in TB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(verify_cache(Path(args.cache_root), args.expected_files, args.min_size_tb))


if __name__ == "__main__":
    main()
