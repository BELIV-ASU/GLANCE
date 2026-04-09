#!/usr/bin/env python3
"""
Prepare reasoning-distillation data from DrivingManual sources.

Inputs:
- California SFT jsonl
- Washington SFT jsonl
- Washington image manifest json

Outputs:
1) teacher_sft_qwq32b.jsonl
   For QwQ-32B QLoRA SFT (messages format)
2) distill_train_qwen25vl7b.jsonl
   For teacher->student distillation (messages + optional text-only image hint)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {idx}") from exc
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_instruction_item(item: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output_text = str(item.get("output", "")).strip()

    user_prompt = instruction
    if input_text:
        user_prompt += "\n\n" + input_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert driving-rule tutor. Give precise, structured, and safety-grounded reasoning."
            ),
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output_text},
    ]

    return {
        "source": source_name,
        "messages": messages,
        "instruction": instruction,
        "input": input_text,
        "target": output_text,
    }


def build_manifest_reasoning_samples(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for m in manifest:
        page = m.get("pdf_page", "unknown")
        description = str(m.get("description", "")).strip()
        image_rel = str(m.get("image_path", "")).strip()

        if not description and not image_rel:
            continue

        prompt = (
            "Given this driving manual visual metadata, infer the likely driving context and safety lesson.\n"
            f"Page: {page}\n"
            f"Description: {description}\n"
            f"Image Path: {image_rel}\n"
            "Provide scenario, conditions, recommended actions, and safety reasoning."
        )

        target = (
            f"Scenario:\nLikely a Washington Driver Guide context around page {page}.\n\n"
            "Conditions:\n"
            f"- Visual cue: {description if description else 'manual illustration'}\n"
            f"- Reference path: {image_rel if image_rel else 'not provided'}\n"
            "- Driver needs rule interpretation and practical behavior guidance\n\n"
            "Actions:\n"
            "- Identify the traffic concept represented in the figure\n"
            "- Cross-check the corresponding handbook rule text\n"
            "- Apply defensive-driving behavior relevant to the scenario\n"
            "- Confirm signage and right-of-way before maneuvering\n\n"
            "Safety Reasoning:\n"
            "Visual handbook elements reinforce recognition speed and correct responses, reducing errors in real traffic situations."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert driving-rule tutor. Produce structured, practical, and safety-first reasoning.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

        samples.append(
            {
                "source": "wa_image_manifest",
                "messages": messages,
                "instruction": "Visual driving-manual reasoning",
                "input": prompt,
                "target": target,
                "image_manifest_path": image_rel,
            }
        )
    return samples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DrivingManual data for reasoning distillation")
    parser.add_argument(
        "--california_jsonl",
        default="/scratch/jnolas77/SafetyVLM/DrivingManual/california_driving_sft/sft_output/california_driving_sft.jsonl",
    )
    parser.add_argument(
        "--wa_jsonl",
        default="/scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/sft_training_data.jsonl",
    )
    parser.add_argument(
        "--image_manifest_json",
        default="/scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/image_manifest.json",
    )
    parser.add_argument(
        "--out_dir",
        default="/scratch/jnolas77/SafetyVLM/reasoning_distillation/data",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    ca_rows = [normalize_instruction_item(x, "california_sft") for x in load_jsonl(Path(args.california_jsonl))]
    wa_rows = [normalize_instruction_item(x, "wa_sft") for x in load_jsonl(Path(args.wa_jsonl))]
    manifest_rows = build_manifest_reasoning_samples(load_json(Path(args.image_manifest_json)))

    all_rows = ca_rows + wa_rows + manifest_rows
    if args.max_samples > 0:
        all_rows = all_rows[: args.max_samples]

    out_dir = Path(args.out_dir)
    teacher_path = out_dir / "teacher_sft_qwq32b.jsonl"
    distill_path = out_dir / "distill_train_qwen25vl7b.jsonl"

    write_jsonl(teacher_path, all_rows)
    write_jsonl(distill_path, all_rows)

    print(f"Prepared {len(all_rows)} records")
    print(f"Teacher SFT file: {teacher_path}")
    print(f"Distillation file: {distill_path}")


if __name__ == "__main__":
    main()
