#!/usr/bin/env python3
"""
Build DriveLM JSONL data for train_teacher.py.

Input:
  - DriveLM scene-level JSON (v1_0_train_nus.json or v1_1_train_nus.json)
Output:
  - <output_dir>/train.jsonl
  - <output_dir>/val.jsonl

Each line is:
  {"messages": [...chat turns...]}
compatible with GLANCE/train_teacher.py.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are DriveLM-Teacher, an expert autonomous-driving reasoning assistant. "
    "Answer grounded in the provided scene context and camera views."
)


def safe_rel_to_abs(data_root: Path, rel_path: str) -> Optional[str]:
    """Resolve a DriveLM relative image path to an absolute path.

    DriveLM JSON uses paths like '../nuscenes/samples/CAM_FRONT/xxx.jpg'
    assuming the JSON lives in  DriveLM/data/QA_dataset_nus/.  When the
    JSON sits at the DriveLM root we need to strip the leading '..' and
    try several candidate locations.
    """
    if not rel_path:
        return None

    # Try 1: direct resolution from data_root
    candidate = (data_root / rel_path).resolve()
    if candidate.is_file():
        return str(candidate)

    # Try 2: strip leading '../' segments (common when JSON is at repo root)
    cleaned = rel_path
    while cleaned.startswith('../'):
        cleaned = cleaned[3:]
    candidate = (data_root / cleaned).resolve()
    if candidate.is_file():
        return str(candidate)

    return None


def build_user_text(
    scene_token: str,
    frame_token: str,
    scene_description: str,
    task: str,
    question: str,
) -> str:
    return (
        f"Scene token: {scene_token}\n"
        f"Frame token: {frame_token}\n"
        f"Task: {task}\n"
        f"Scene description: {scene_description}\n\n"
        f"Question: {question}"
    )


def iter_qa_samples(data: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for scene_token, scene_obj in data.items():
        scene_description = scene_obj.get("scene_description", "")
        key_frames = scene_obj.get("key_frames", {}) or {}

        for frame_token, frame_obj in key_frames.items():
            qa_block = frame_obj.get("QA", {}) or {}
            image_paths = frame_obj.get("image_paths", {}) or {}

            for task_name, qa_list in qa_block.items():
                if not isinstance(qa_list, list):
                    continue

                for qa in qa_list:
                    q = (qa.get("Q") or "").strip()
                    a = (qa.get("A") or "").strip()
                    if not q or not a:
                        continue

                    rows.append(
                        {
                            "scene_token": scene_token,
                            "frame_token": frame_token,
                            "scene_description": scene_description,
                            "task": task_name,
                            "question": q,
                            "answer": a,
                            "image_paths": image_paths,
                        }
                    )
    return rows


def _derive_depth_path(abs_img: str) -> Optional[str]:
    """Derive the pre-computed depth map path from an RGB image path.

    Mapping: .../nuscenes/samples/CAM_FRONT/xxx.jpg
           → .../nuscenes/depth/CAM_FRONT/xxx.png
    """
    p = Path(abs_img)
    # Replace 'samples' directory with 'depth' and .jpg → .png
    parts = list(p.parts)
    try:
        idx = parts.index("samples")
        parts[idx] = "depth"
    except ValueError:
        return None
    depth_p = Path(*parts).with_suffix(".png")
    if depth_p.is_file():
        return str(depth_p)
    return None


def to_conversation(
    row: Dict,
    data_root: Path,
    camera: str,
    with_images: bool,
    with_depth: bool = False,
) -> Dict:
    user_text = build_user_text(
        scene_token=row["scene_token"],
        frame_token=row["frame_token"],
        scene_description=row["scene_description"],
        task=row["task"],
        question=row["question"],
    )

    user_content: List[Dict] = [{"type": "text", "text": user_text}]

    if with_images:
        rel_img = row.get("image_paths", {}).get(camera, "")
        abs_img = safe_rel_to_abs(data_root, rel_img)
        if abs_img:
            user_content.append({"type": "image", "image": f"file://{abs_img}"})

            # Optionally add pre-computed depth map as a second image
            if with_depth:
                depth_path = _derive_depth_path(abs_img)
                if depth_path:
                    user_content.append(
                        {"type": "image", "image": f"file://{depth_path}"}
                    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"messages": messages}


def split_dataset(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    random.shuffle(rows)

    if not rows:
        return [], []

    n_val = max(1, int(len(rows) * val_ratio))
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


def save_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM train/val JSONL for GLANCE trainer")
    parser.add_argument("--data_json", required=True, help="Path to DriveLM v1_x_train_nus.json")
    parser.add_argument("--data_root", required=True, help="DriveLM root directory")
    parser.add_argument("--output_dir", required=True, help="Directory to write train.jsonl and val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera key from image_paths")
    parser.add_argument("--no_images", action="store_true", help="Build text-only dataset")
    parser.add_argument("--with_depth", action="store_true",
                        help="Include pre-computed depth maps as a second image")
    args = parser.parse_args()

    data_json = Path(args.data_json)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    with data_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = iter_qa_samples(raw)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    train_rows, val_rows = split_dataset(rows, val_ratio=args.val_ratio, seed=args.seed)

    train_convs = [to_conversation(r, data_root, args.camera, with_images=not args.no_images, with_depth=args.with_depth) for r in train_rows]
    val_convs = [to_conversation(r, data_root, args.camera, with_images=not args.no_images, with_depth=args.with_depth) for r in val_rows]

    save_jsonl(train_convs, output_dir / "train.jsonl")
    save_jsonl(val_convs, output_dir / "val.jsonl")

    print(f"Loaded QA samples: {len(rows)}")
    print(f"Train samples: {len(train_convs)}")
    print(f"Val samples:   {len(val_convs)}")
    print(f"Output dir:    {output_dir}")


if __name__ == "__main__":
    main()
