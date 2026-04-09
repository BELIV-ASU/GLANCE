#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq as VisionModelClass
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VisionModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VisionModelClass


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx} in {path}") from exc
    return items


def maybe_image(path: Optional[str]) -> Optional[Image.Image]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def build_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in item:
        return item["messages"]
    if "text" in item:
        return [{"role": "user", "content": [{"type": "text", "text": item["text"]}]}]
    raise KeyError("Sample must include 'messages' or 'text'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate distilled Qwen VL student")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all rows")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = VisionModelClass.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(iter(model.parameters())).device

    data = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as w:
        for i, item in enumerate(data):
            messages = build_messages(item)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image = maybe_image(item.get("image"))

            if image is None:
                inputs = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                inputs = processor(
                    text=[prompt],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
            }
            if args.temperature > 0.0:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[:, prompt_len:]
            prediction = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            row = {
                "index": i,
                "prediction": prediction,
                "target": item.get("target"),
                "image": item.get("image"),
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
