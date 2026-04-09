#!/usr/bin/env python3
"""
Build GLANCE/JSONL training data that fuses:
  1. Annotated Existing Driving Data  (ground-truth meta-actions  → S1–S4)
  2. Annotated Counterfactual Data    (perturbed meta-actions     → S1–S4)

Four-step reasoning chain per paper (Fig. 3):
  S1  Scene Understanding        (shared between both branches)
  S2  Crucial Object Detection   (shared between both branches)
  S3  Behavior Prediction        (GT actions for existing data;
                                   perturbed/counterfactual actions for CF data)
  S4  Safety Analysis            (jointly supervised by both datasets)

Input:
  --data_json   DriveLM scene-level JSON  (v1_0_train_nus.json …)
  --cf_json     (optional) counterfactual JSONL produced by cf_gen.py
  --data_root   Repo / nuscenes root for resolving image paths

Output:
  <output_dir>/train.jsonl
  <output_dir>/val.jsonl

Each line: {"messages": [...chat turns...], "source": "existing"|"counterfactual"}
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a safety-aware autonomous driving reasoning assistant. "
    "Given annotated driving frames with tracked object IDs, analyse the "
    "scene step-by-step: understand the environment (S1), identify crucial "
    "dynamic objects (S2), predict their behaviour from meta-actions (S3), "
    "and evaluate overall driving safety (S4)."
)

# Chain-of-thought step headers (match Fig. 3 of the paper)
_S1 = "**S1 – Scene Understanding:**"
_S2 = "**S2 – Crucial Object Detection:**"
_S3 = "**S3 – Behavior Prediction:**"
_S4 = "**S4 – Safety Analysis:**"


# ---------------------------------------------------------------------------
# Image-path helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def safe_rel_to_abs(data_root: Path, rel_path: str) -> Optional[str]:
    """Resolve a DriveLM relative image path to an absolute path.

    DriveLM JSON uses paths like '../nuscenes/samples/CAM_FRONT/xxx.jpg'
    assuming the JSON lives in DriveLM/data/QA_dataset_nus/.  When the
    JSON sits at the DriveLM root we need to strip the leading '..' and
    try several candidate locations.
    """
    if not rel_path:
        return None

    candidate = (data_root / rel_path).resolve()
    if candidate.is_file():
        return str(candidate)

    cleaned = rel_path
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    candidate = (data_root / cleaned).resolve()
    if candidate.is_file():
        return str(candidate)

    return None


def _derive_depth_path(abs_img: str) -> Optional[str]:
    """Derive the pre-computed depth map path from an RGB image path.

    Mapping: .../nuscenes/samples/CAM_FRONT/xxx.jpg
           → .../nuscenes/depth/CAM_FRONT/xxx.png
    """
    p = Path(abs_img)
    parts = list(p.parts)
    try:
        idx = parts.index("samples")
        parts[idx] = "depth"
    except ValueError:
        return None
    depth_p = Path(*parts).with_suffix(".png")
    return str(depth_p) if depth_p.is_file() else None


# ---------------------------------------------------------------------------
# Existing DriveLM data  →  flat row list
# ---------------------------------------------------------------------------

def iter_existing_samples(data: Dict) -> List[Dict]:
    """Flatten DriveLM JSON into per-QA rows enriched with safety metadata.

    New fields relative to the original script:
      ego_action          – ground-truth ego meta-action string
      agent_actions       – {object_id: meta_action_str, …}
      safety_label        – "safe" | "unsafe" | None
      scene_understanding – free-text S1 (if stored in QA entry)
      crucial_objects     – free-text S2 (if stored in QA entry)
    """
    rows: List[Dict] = []
    for scene_token, scene_obj in data.items():
        scene_description = scene_obj.get("scene_description", "")
        key_frames = scene_obj.get("key_frames", {}) or {}

        for frame_token, frame_obj in key_frames.items():
            qa_block    = frame_obj.get("QA", {}) or {}
            image_paths = frame_obj.get("image_paths", {}) or {}

            # Ground-truth meta-actions and safety label from frame metadata
            ego_action    = frame_obj.get("ego_meta_action", "")
            agent_actions = frame_obj.get("agent_meta_actions", {}) or {}
            safety_label  = frame_obj.get("safety_label", None)

            for task_name, qa_list in qa_block.items():
                if not isinstance(qa_list, list):
                    continue

                for qa in qa_list:
                    q = (qa.get("Q") or "").strip()
                    a = (qa.get("A") or "").strip()
                    if not q or not a:
                        continue

                    rows.append({
                        "source":             "existing",
                        "scene_token":        scene_token,
                        "frame_token":        frame_token,
                        "scene_description":  scene_description,
                        "task":               task_name,
                        "question":           q,
                        "answer":             a,
                        "image_paths":        image_paths,
                        # S3 inputs
                        "ego_action":         ego_action,
                        "agent_actions":      agent_actions,
                        # S4 label
                        "safety_label":       safety_label,
                        # Optional pre-computed S1/S2 text stored alongside QA
                        "scene_understanding": qa.get("scene_understanding", ""),
                        "crucial_objects":     qa.get("crucial_objects", ""),
                    })
    return rows


# ---------------------------------------------------------------------------
# Counterfactual data  →  flat row list
# ---------------------------------------------------------------------------

def iter_counterfactual_samples(cf_path: Path) -> List[Dict]:
    """Load counterfactual JSONL produced by cf_gen.py.

    Expected fields per line (paper §III-A):
      scene_token          – matches parent DriveLM scene
      frame_token          – last frame of the parent clip  (I˜_τ)
      image_paths          – same camera dict as existing data
      ego_cf_action        – perturbed ego meta-action, e.g. "keep lane, accelerate"
      agent_cf_actions     – {object_id: meta_action, …}
      safety_label         – "safe" | "unsafe"
      scene_description    – (optional) copied from parent clip
      scene_understanding  – (optional) shared S1 text from parent clip
      crucial_objects      – (optional) shared S2 text from parent clip

    Fields are normalised to match the schema used for existing rows so the
    same `to_conversation` builder works for both sources.
    """
    rows: List[Dict] = []
    with cf_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Normalise CF-specific field names → shared schema
            obj["source"]        = "counterfactual"
            obj["ego_action"]    = obj.pop("ego_cf_action",    obj.get("ego_action", ""))
            obj["agent_actions"] = obj.pop("agent_cf_actions", obj.get("agent_actions", {}))

            # Default question / answer for CF rows that have no explicit QA
            obj.setdefault("task",     "safety_prediction")
            obj.setdefault("question",
                           "Given the predicted meta-actions, is this scenario safe or unsafe?")
            # The binary label is the canonical S4 answer for CF rows
            obj.setdefault("answer", obj.get("safety_label", "safe"))

            # Fallbacks for shared S1/S2 fields
            obj.setdefault("scene_description",  "")
            obj.setdefault("scene_understanding", "")
            obj.setdefault("crucial_objects",     "")

            rows.append(obj)
    return rows

def _format_s3_behavior(ego_action: str, agent_actions: Dict) -> str:
    """Format S3 Behavior Prediction block.

    For existing data this uses ground-truth meta-actions.
    For counterfactual data this uses the perturbed meta-actions.
    Both are stored under the same keys after normalisation in the loaders.
    """
    lines: List[str] = []
    if ego_action:
        lines.append(f"Ego: {ego_action}")
    for obj_id, action in (agent_actions or {}).items():
        lines.append(f"Object {obj_id}: {action}")
    return "\n".join(lines) if lines else "No action metadata available."


def _format_s4_safety(safety_label: Optional[str], answer: str) -> str:
    """Format S4 Safety Analysis block.

    Prefer the free-text answer (existing QA data); fall back to a minimal
    verdict string derived from the binary safety label (counterfactual data).
    """
    if answer:
        return answer
    if safety_label:
        return f"Safety verdict: {safety_label.capitalize()}."
    return "Safety verdict: Unknown."


def build_cot_answer(row: Dict) -> str:
    """Compose the full S1–S4 chain-of-thought assistant turn.

    Paper Fig. 3 supervision logic:
      • S1 (Scene Understanding)      – supervised by existing data; shared with CF.
      • S2 (Crucial Object Detection) – supervised by existing data; shared with CF.
      • S3 (Behavior Prediction)      – GT actions for existing; CF actions for CF rows.
      • S4 (Safety Analysis)          – jointly supervised by both datasets.
    """
    s1_text = row.get("scene_understanding") or row.get("scene_description", "")
    s2_text = row.get("crucial_objects", "")
    s3_text = _format_s3_behavior(row.get("ego_action", ""), row.get("agent_actions", {}))
    s4_text = _format_s4_safety(row.get("safety_label"), row.get("answer", ""))

    parts: List[str] = []
    if s1_text:
        parts.append(f"{_S1}\n{s1_text}")
    if s2_text:
        parts.append(f"{_S2}\n{s2_text}")
    parts.append(f"{_S3}\n{s3_text}")
    parts.append(f"{_S4}\n{s4_text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------

def _build_user_text(row: Dict) -> str:
    lines = [
        f"Scene token:  {row['scene_token']}",
        f"Frame token:  {row['frame_token']}",
        f"Task:         {row['task']}",
    ]
    if row.get("scene_description"):
        lines.append(f"Scene description: {row['scene_description']}")
    if row.get("source") == "counterfactual":
        lines.append("[Counterfactual scenario – meta-actions are perturbed]")
    lines.append("")
    lines.append(f"Question: {row['question']}")
    return "\n".join(lines)


def to_conversation(
    row: Dict,
    data_root: Path,
    camera: str,
    with_images: bool,
    with_depth: bool = False,
    use_cot: bool = True,
) -> Dict:
    """Build a single {"messages": [...]} training example.

    Args:
        row:         Flat sample dict (existing or counterfactual).
        data_root:   Root directory for resolving image paths.
        camera:      Camera key, e.g. "CAM_FRONT".
        with_images: Whether to attach image content blocks.
        with_depth:  Also attach a pre-computed depth map as a second image.
        use_cot:     Emit full S1–S4 chain-of-thought; if False, raw answer only.
    """
    user_content: List[Dict] = [{"type": "text", "text": _build_user_text(row)}]

    if with_images:
        rel_img = row.get("image_paths", {}).get(camera, "")
        abs_img = safe_rel_to_abs(data_root, rel_img)
        if abs_img:
            user_content.append({"type": "image", "image": f"file://{abs_img}"})
            if with_depth:
                depth_path = _derive_depth_path(abs_img)
                if depth_path:
                    user_content.append({"type": "image", "image": f"file://{depth_path}"})

    assistant_answer = build_cot_answer(row) if use_cot else row.get("answer", "")

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_answer},
    ]
    return {"messages": messages, "source": row.get("source", "existing")}


# ---------------------------------------------------------------------------
# Train / val split + JSONL writer
# ---------------------------------------------------------------------------

def split_dataset(
    rows: List[Dict], val_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    random.shuffle(rows)
    if not rows:
        return [], []
    n_val = max(1, int(len(rows) * val_ratio))
    val   = rows[:n_val]
    train = rows[n_val:]
    return train, val


def save_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build fused DriveLM + counterfactual JSONL for GLANCE trainer. "
            "Produces S1–S4 chain-of-thought examples per the paper (Fig. 3)."
        )
    )
    # --- Input ---
    parser.add_argument("--data_json", required=True,
                        help="Path to DriveLM v1_x_train_nus.json")
    parser.add_argument("--cf_json",   default="",
                        help="(optional) Counterfactual JSONL from cf_gen.py")
    parser.add_argument("--data_root", required=True,
                        help="DriveLM / nuscenes root for resolving image paths")
    # --- Output ---
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write train.jsonl and val.jsonl")
    # --- Split ---
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--max_samples", type=int,   default=0,
                        help="Cap total samples (0 = use all)")
    # --- Vision ---
    parser.add_argument("--camera",     default="CAM_FRONT",
                        help="Camera key from image_paths dict")
    parser.add_argument("--no_images",  action="store_true",
                        help="Build text-only dataset (no image content blocks)")
    parser.add_argument("--with_depth", action="store_true",
                        help="Include pre-computed depth maps as a second image")
    # --- Reasoning ---
    parser.add_argument("--no_cot",     action="store_true",
                        help="Skip S1–S4 chain-of-thought; emit raw answer only")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # --- Load existing DriveLM QA data ---
    with Path(args.data_json).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    existing_rows = iter_existing_samples(raw)
    print(f"Existing QA samples    : {len(existing_rows)}")

    # --- Load counterfactual data (optional) ---
    cf_rows: List[Dict] = []
    if args.cf_json:
        cf_rows = iter_counterfactual_samples(Path(args.cf_json))
        print(f"Counterfactual samples : {len(cf_rows)}")

    # --- Fuse and optionally cap ---
    all_rows = existing_rows + cf_rows
    if args.max_samples > 0:
        all_rows = all_rows[: args.max_samples]
    print(f"Total samples          : {len(all_rows)}")

    # --- Split ---
    train_rows, val_rows = split_dataset(all_rows, args.val_ratio, args.seed)

    # --- Build conversations ---
    conv_kwargs = dict(
        data_root   = data_root,
        camera      = args.camera,
        with_images = not args.no_images,
        with_depth  = args.with_depth,
        use_cot     = not args.no_cot,
    )
    train_convs = [to_conversation(r, **conv_kwargs) for r in train_rows]
    val_convs   = [to_conversation(r, **conv_kwargs) for r in val_rows]

    # --- Save ---
    save_jsonl(train_convs, output_dir / "train.jsonl")
    save_jsonl(val_convs,   output_dir / "val.jsonl")

    # --- Report ---
    n_cf_train = sum(1 for r in train_rows if r.get("source") == "counterfactual")
    n_cf_val   = sum(1 for r in val_rows   if r.get("source") == "counterfactual")
    print(f"Train  : {len(train_convs):>7,}  (counterfactual: {n_cf_train:,})")
    print(f"Val    : {len(val_convs):>7,}  (counterfactual: {n_cf_val:,})")
    print(f"Output : {output_dir}")


if __name__ == "__main__":
    main()