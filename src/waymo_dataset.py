#!/usr/bin/env python3
"""
waymo_dataset.py  –  PyTorch Dataset for Waymo image frames + text prompts
═══════════════════════════════════════════════════════════════════════════════

Reads Waymo camera images (extracted JPEGs or raw .tfrecord) and pairs them
with text prompts / annotations for VLM fine-tuning on traffic understanding.

Two ingestion paths are supported:
  1. **Extracted images** (fast) – pre-extracted JPEGs in a directory tree,
     paired with an annotations JSON file.
  2. **TFRecord**  (raw)  – reads Waymo .tfrecord files directly via the
     `tensorflow` / `waymo-open-dataset` SDK.  Requires `pip install
     waymo-open-dataset-tf-2-12-0`.

The Dataset is processor-agnostic: it returns raw PIL images + text strings.
The `WaymoCollator` wraps a VLM-specific processor to produce padded/batched
tensors at collation time — so switching VLMs only requires swapping the
processor, **not** the dataset.

Usage:
    from waymo_dataset import WaymoImageDataset, WaymoCollator
    dataset = WaymoImageDataset(cfg["data"])
    collator = WaymoCollator(processor, max_length=2048)
    loader  = DataLoader(dataset, batch_size=2, collate_fn=collator)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Annotation schema
# ═══════════════════════════════════════════════════════════════════════════
#
#  The expected annotations JSON is a list of objects:
#
#  [
#    {
#      "image_path": "training/segment-xxx/camera_FRONT/000.jpg",
#      "camera":     "FRONT",
#      "prompt":     "Describe any traffic violations visible in this frame.",
#      "answer":     "A sedan is running a red light at the intersection ...",
#      "metadata": { ... }          // optional, ignored during training
#    },
#    ...
#  ]
#
#  `image_path` is relative to `waymo_root`.
#  When annotations don't exist yet, a *placeholder* mode synthesises
#  dummy prompts so the training loop can be dry-run.
# ═══════════════════════════════════════════════════════════════════════════


PLACEHOLDER_PROMPT = (
    "Describe the traffic scenario in this image. "
    "Identify any vehicles, pedestrians, traffic signals, and potential violations."
)
PLACEHOLDER_ANSWER = (
    "This is a placeholder answer. Replace with actual Waymo annotations."
)


# ---------------------------------------------------------------------------
#  Annotation loading helpers
# ---------------------------------------------------------------------------

def load_annotations(json_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """
    Load the annotation file.

    Parameters
    ----------
    json_path : str
        Path to the annotations JSON (list-of-dicts).
    max_samples : int
        If > 0, truncate to this many samples (useful for debug runs).

    Returns
    -------
    list[dict]
    """
    if not json_path or not os.path.isfile(json_path):
        log.warning(
            "Annotations file not found (%s). "
            "Falling back to placeholder mode.", json_path
        )
        return []

    path = Path(json_path)
    if path.suffix.lower() == ".jsonl":
        data: List[Dict[str, Any]] = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

    if max_samples > 0:
        data = data[:max_samples]

    log.info("Loaded %d annotations from %s", len(data), json_path)
    return data


def discover_images(waymo_root: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """
    Walk `waymo_root` and build placeholder annotations for every image found.

    This is the fallback when no curated annotation file exists yet.
    Useful for testing the pipeline end-to-end.
    """
    root = Path(waymo_root)
    extensions = {".jpg", ".jpeg", ".png"}
    samples: List[Dict[str, Any]] = []

    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() in extensions:
            samples.append({
                "image_path": str(img_path.relative_to(root)),
                "prompt": PLACEHOLDER_PROMPT,
                "answer": PLACEHOLDER_ANSWER,
            })
            if max_samples > 0 and len(samples) >= max_samples:
                break

    log.info(
        "Discovered %d images under %s (placeholder mode)", len(samples), waymo_root
    )
    return samples


# ---------------------------------------------------------------------------
#  TFRecord reader (optional – only needed when working with raw Waymo data)
# ---------------------------------------------------------------------------

def load_from_tfrecords(
    tfrecord_dir: str,
    camera_name: str = "FRONT",
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """
    Parse Waymo .tfrecord files directly and return annotation dicts with
    in-memory images.

    ⚠️  Requires:
        pip install tensorflow waymo-open-dataset-tf-2-12-0

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing *.tfrecord files.
    camera_name : str
        Which camera to extract ("FRONT", "FRONT_LEFT", "FRONT_RIGHT",
        "SIDE_LEFT", "SIDE_RIGHT").
    max_samples : int
        If > 0, stop after this many frames.

    Returns
    -------
    list[dict]  with keys: "pil_image", "prompt", "answer", "camera"
    """
    # ── Lazy imports (TF + Waymo SDK are heavy) ──────────────────────────
    try:
        import tensorflow as tf                         # noqa: F401
        from waymo_open_dataset import dataset_pb2      # noqa: F401
        from waymo_open_dataset.utils import frame_utils  # noqa: F401
    except ImportError:
        log.error(
            "TFRecord ingestion requires `tensorflow` and "
            "`waymo-open-dataset-tf-2-12-0`.  Install them or use "
            "pre-extracted images instead."
        )
        return []

    import io

    # ── Camera name → enum value mapping ─────────────────────────────────
    camera_enum = {
        "FRONT":       1,
        "FRONT_LEFT":  2,
        "FRONT_RIGHT": 3,
        "SIDE_LEFT":   4,
        "SIDE_RIGHT":  5,
    }
    cam_id = camera_enum.get(camera_name.upper(), 1)

    samples: List[Dict[str, Any]] = []
    tfrecord_files = sorted(Path(tfrecord_dir).glob("*.tfrecord"))
    log.info("Found %d tfrecord files in %s", len(tfrecord_files), tfrecord_dir)

    for tf_path in tfrecord_files:
        dataset = tf.data.TFRecordDataset(str(tf_path), compression_type="")
        for raw_record in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(raw_record.numpy())

            for cam_image in frame.images:
                if cam_image.name != cam_id:
                    continue

                # ── Decode the JPEG ──────────────────────────────────────
                pil_img = Image.open(io.BytesIO(cam_image.image)).convert("RGB")
                samples.append({
                    "pil_image": pil_img,
                    "camera": camera_name,
                    "prompt": PLACEHOLDER_PROMPT,
                    "answer": PLACEHOLDER_ANSWER,
                })

                if max_samples > 0 and len(samples) >= max_samples:
                    log.info("Reached max_samples=%d from tfrecords", max_samples)
                    return samples

    log.info("Loaded %d frames from tfrecords (camera=%s)", len(samples), camera_name)
    return samples


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

class WaymoImageDataset(Dataset):
    """
    PyTorch Dataset for Waymo images + text prompts.

    Returns
    -------
    dict with keys:
        "image"  : PIL.Image.Image   (RGB, original resolution)
        "prompt" : str                (user question / instruction)
        "answer" : str                (target completion)
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split: str = "train",
    ):
        """
        Parameters
        ----------
        data_cfg : dict
            The `data:` section of waymo_finetune_config.yaml.
        split : str
            "train" or "val".
        """
        super().__init__()
        self.waymo_root = Path(data_cfg["waymo_root"])
        self.image_size: Tuple[int, int] = tuple(data_cfg.get("image_size", [448, 448]))
        max_samples: int = data_cfg.get("max_samples", 0)

        # ── Pick annotation file by split ────────────────────────────────
        if split == "val":
            ann_path = data_cfg.get("val_annotations_json", "")
        else:
            ann_path = data_cfg.get("annotations_json", "")

        # ── Load annotations (or fall back to discovery / tfrecord) ──────
        self.samples = load_annotations(ann_path, max_samples)

        if not self.samples:
            # Try auto-discovering images
            self.samples = discover_images(str(self.waymo_root), max_samples)

        if not self.samples:
            log.warning(
                "No data found for split=%s.  The dataset will be empty. "
                "Provide an annotations JSON or place images under %s.",
                split, self.waymo_root,
            )

        # ── Flag: some samples may carry in-memory PIL images (tfrecord) ─
        self._has_inline_images = (
            len(self.samples) > 0 and "pil_image" in self.samples[0]
        )

        log.info(
            "WaymoImageDataset(%s): %d samples, image_size=%s",
            split, len(self.samples), self.image_size,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # ── Load image ───────────────────────────────────────────────────
        if self._has_inline_images:
            image = sample["pil_image"]
        else:
            img_path = self.waymo_root / sample["image_path"]
            if not img_path.exists():
                # Graceful fallback: return a black placeholder
                log.warning("Image not found: %s", img_path)
                image = Image.new("RGB", self.image_size, (0, 0, 0))
            else:
                image = Image.open(img_path).convert("RGB")

        # ── Resize to target dimensions ──────────────────────────────────
        # TODO: Some VLMs (Qwen2.5-VL) handle dynamic resolution natively.
        #       You may want to skip this resize for those models and let the
        #       processor handle it.  For now we resize uniformly.
        if self.image_size:
            image = image.resize(
                (self.image_size[0], self.image_size[1]),
                Image.LANCZOS,
            )

        return {
            "image":  image,
            "prompt": sample.get("prompt", PLACEHOLDER_PROMPT),
            "answer": sample.get("answer", PLACEHOLDER_ANSWER),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Collator (processor-aware batching)
# ═══════════════════════════════════════════════════════════════════════════

class WaymoCollator:
    """
    Collate function that wraps a VLM-specific processor.

    Handles:
      • Dynamic image sizes  (each sample can have a different resolution)
      • Text tokenisation + padding
      • Label masking        (loss computed only on the answer tokens)

    Usage:
        collator = WaymoCollator(processor, max_length=2048, model_name="qwen2.5-vl-7b")
        loader = DataLoader(dataset, collate_fn=collator, batch_size=2)
    """

    def __init__(
        self,
        processor: Any,
        max_length: int = 2048,
        model_name: str = "qwen2.5-vl-7b",
        padding: str = "longest",
    ):
        self.processor = processor
        self.max_length = max_length
        self.model_name = model_name.lower()
        self.padding = padding

    # ------------------------------------------------------------------

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : list[dict]
            Each dict has keys: "image" (PIL), "prompt" (str), "answer" (str)

        Returns
        -------
        dict[str, Tensor]
            Ready for `model(**batch)`.  Includes `labels` with prompt tokens
            masked to -100.
        """
        images:  List[Image.Image] = [s["image"] for s in batch]
        prompts: List[str]         = [s["prompt"] for s in batch]
        answers: List[str]         = [s["answer"] for s in batch]

        # ── Dispatch to model-specific formatting ────────────────────────
        if "qwen" in self.model_name:
            return self._collate_qwen(images, prompts, answers)
        elif "openvla" in self.model_name:
            return self._collate_openvla(images, prompts, answers)
        elif "vila" in self.model_name:
            return self._collate_vila(images, prompts, answers)
        else:
            # Generic fallback — works for most HF VLMs with a .processor
            return self._collate_generic(images, prompts, answers)

    # ── Qwen2.5-VL  ──────────────────────────────────────────────────────

    def _collate_qwen(
        self,
        images: List[Image.Image],
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Qwen2.5-VL expects chat-template messages with image placeholders.

        Each sample becomes:
          [
            {"role": "user",      "content": [
                {"type": "image"},
                {"type": "text", "text": <prompt>}
            ]},
            {"role": "assistant", "content": <answer>},
          ]
        """
        conversations = []
        for prompt, answer in zip(prompts, answers):
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ])

        # Apply the chat template to get the full text
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False,
            )
            for conv in conversations
        ]

        # Processor handles image encoding + text tokenisation together
        batch_encoding = self.processor(
            text=texts,
            images=images,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ── Build labels (mask everything before the assistant answer) ───
        labels = batch_encoding["input_ids"].clone()

        # TODO: Implement precise label masking.
        # For now, use a simple heuristic: find the assistant header token
        # and mask everything before it to -100.
        #
        # A proper implementation should:
        #   1. Tokenise the prompt-only text to find its length.
        #   2. Set labels[:prompt_len] = -100
        #   3. Also mask padding tokens.
        #
        # Placeholder: mask pad tokens only
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch_encoding["labels"] = labels
        return batch_encoding

    # ── OpenVLA  ──────────────────────────────────────────────────────────

    def _collate_openvla(
        self,
        images: List[Image.Image],
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        OpenVLA uses a Prismatic-style processor.

        TODO: Adapt to OpenVLA's exact input format once the model is integrated.
              OpenVLA typically expects:
                processor(prompt_text, image) → {"pixel_values", "input_ids", ...}
        """
        full_texts = [f"{p}\n{a}" for p, a in zip(prompts, answers)]

        batch_encoding = self.processor(
            text=full_texts,
            images=images,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch_encoding["input_ids"].clone()
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch_encoding["labels"] = labels
        return batch_encoding

    # ── VILA  ─────────────────────────────────────────────────────────────

    def _collate_vila(
        self,
        images: List[Image.Image],
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        VILA uses LLaVA-style conversation.

        TODO: Adapt to VILA-1.5 exact processor API.
              VILA typically expects:
                conversation with <image> token in the text.
        """
        full_texts = [f"<image>\n{p}\n{a}" for p, a in zip(prompts, answers)]

        batch_encoding = self.processor(
            text=full_texts,
            images=images,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch_encoding["input_ids"].clone()
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch_encoding["labels"] = labels
        return batch_encoding

    # ── Generic fallback  ────────────────────────────────────────────────

    def _collate_generic(
        self,
        images: List[Image.Image],
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Generic collation for VLMs with a standard HF processor.

        Works for models that accept:
            processor(text=..., images=..., return_tensors="pt")
        """
        full_texts = [f"{p}\n{a}" for p, a in zip(prompts, answers)]

        batch_encoding = self.processor(
            text=full_texts,
            images=images,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch_encoding["input_ids"].clone()
        # Mask padding
        if hasattr(self.processor, "tokenizer"):
            pad_id = self.processor.tokenizer.pad_token_id
        elif hasattr(self.processor, "pad_token_id"):
            pad_id = self.processor.pad_token_id
        else:
            pad_id = None

        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch_encoding["labels"] = labels
        return batch_encoding


# ═══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    # Load config
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "waymo_finetune_config.yaml"
    )
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "data": {
                "waymo_root": "/scratch/rbaskar5/Dataset/waymo",
                "annotations_json": "",
                "image_size": [448, 448],
                "max_samples": 5,
            }
        }

    ds = WaymoImageDataset(cfg["data"], split="train")
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"  image size:  {sample['image'].size}")
        print(f"  prompt:      {sample['prompt'][:80]}...")
        print(f"  answer:      {sample['answer'][:80]}...")
