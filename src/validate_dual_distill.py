"""
validate_dual_distill.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates the dual-distilled Qwen2.5-VL-7B student adapter on:

  • Waymo front-camera trajectory prediction
      metrics: loss, perplexity, ADE (m), FDE (m), waypoint parse rate
  • Reasoning (text-only)
      metrics: loss, perplexity, ROUGE-L

Val data is sampled deterministically from train.jsonl (--val_fraction, default 10%).
If you later get dedicated val splits pass --waymo_val_jsonl / --reason_val_jsonl.

Outputs (in --out_dir):
  metrics.json           ← aggregate numbers for all evaluated checkpoints
  samples_waymo.csv      ← per-sample results (loss, ADE, FDE, generated text)
  samples_reasoning.csv  ← per-sample results (loss, ROUGE-L, generated text)

Usage
─────
python validate_dual_distill.py \
  --adapter_path /scratch/rbaskar5/GLANCE/checkpoints/dual_distilled_7b/checkpoint-200 \
  --waymo_jsonl  /scratch/rbaskar5/GLANCE/data/waymo_trajectory/train.jsonl \
  --reason_jsonl /scratch/rbaskar5/GLANCE/reasoning_distillation/data/distill_train_qwen25vl7b.jsonl \
  --image_root   /scratch/rbaskar5/Dataset/waymo_front \
  --out_dir      /scratch/rbaskar5/GLANCE/evals/ckpt200

# To compare multiple checkpoints in one run, pass multiple --adapter_path values:
#   --adapter_path .../checkpoint-200 .../final
# Results for each are appended to metrics.json.

# To skip text generation (faster, loss/ppl only):
#   --skip_generation
"""

import argparse, csv, json, math, os, re, sys, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# ── optional rouge-score ─────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[warn] rouge-score not found. pip install rouge-score for ROUGE-L.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ARGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HF base model id (same as training)")
    p.add_argument("--adapter_path", nargs="+", required=True,
                   help="Path(s) to LoRA adapter dir(s). Evaluated one by one.")

    # data — train splits (val sampled from these)
    p.add_argument("--waymo_jsonl",  required=True)
    p.add_argument("--reason_jsonl", required=True)
    p.add_argument("--image_root",   required=True,
                   help="Root dir containing Waymo front camera images")

    # optional: dedicated val splits (override sampling)
    p.add_argument("--waymo_val_jsonl",  default=None)
    p.add_argument("--reason_val_jsonl", default=None)

    # val sampling
    p.add_argument("--val_fraction", type=float, default=0.10,
                   help="Fraction of train used as val (ignored if val jsonl provided)")
    p.add_argument("--val_seed",     type=int,   default=42)
    p.add_argument("--max_val_waymo",   type=int, default=200,
                   help="Cap Waymo val samples (generation is slow)")
    p.add_argument("--max_val_reason",  type=int, default=200)

    # generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--skip_generation", action="store_true",
                   help="Only compute loss/perplexity; no text generation (fast mode)")

    # hardware
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype",  default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # output
    p.add_argument("--out_dir", required=True)

    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. DATA UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_val(records: List[dict], fraction: float, seed: int,
               cap: int) -> List[dict]:
    """
    Deterministic val split — shuffles with fixed seed, takes first `fraction`
    of the shuffled list (so train/val never overlap across runs).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records)).tolist()
    n   = min(cap, max(1, int(len(records) * fraction)))
    val_idx = sorted(idx[:n])
    return [records[i] for i in val_idx]


def extract_conversations(rec: dict) -> Tuple[str, str]:
    """Return (user_prompt, assistant_ground_truth) from a record."""
    user, assistant = "", ""
    for turn in rec.get("conversations", []):
        role = turn.get("role", "")
        if role == "user":
            user = turn.get("content", "")
        elif role == "assistant":
            assistant = turn.get("content", "")
    return user, assistant


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. WAYPOINT PARSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matches patterns like:
#   (1.23, 4.56)   [1.23, 4.56]   1.23, 4.56   x=1.23 y=4.56
WAYPOINT_RE = re.compile(
    r"[\[\(]?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]?",
    re.IGNORECASE
)

def parse_waypoints(text: str) -> Optional[np.ndarray]:
    """
    Extract (x, y) waypoints from model output text.
    Returns np.ndarray of shape (N, 2) or None if nothing parseable found.
    Points that look like image coordinates (both >100) are filtered out.
    """
    matches = WAYPOINT_RE.findall(text)
    if not matches:
        return None
    pts = []
    for x_str, y_str in matches:
        x, y = float(x_str), float(y_str)
        # Heuristic: Waymo BEV coords are typically within ±80 m
        if abs(x) < 150 and abs(y) < 150:
            pts.append([x, y])
    return np.array(pts, dtype=np.float32) if pts else None


def compute_ade_fde(pred: np.ndarray,
                    gt:   np.ndarray) -> Tuple[float, float]:
    """
    ADE = mean L2 distance over all matched waypoints
    FDE = L2 distance at the final waypoint

    If pred and gt have different lengths, we align by taking min(len).
    """
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    pred_aligned = pred[:n]
    gt_aligned   = gt[:n]
    dists = np.linalg.norm(pred_aligned - gt_aligned, axis=1)  # (n,)
    ade   = float(dists.mean())
    fde   = float(dists[-1])
    return ade, fde


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ROUGE-L
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_rouge_l(prediction: str, reference: str) -> float:
    if not HAS_ROUGE:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score  = scorer.score(reference, prediction)
    return round(score["rougeL"].fmeasure, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. MODEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model_and_processor(base_model: str, adapter_path: str,
                              device: str, dtype_str: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[dtype_str]

    print(f"[model] Loading base: {base_model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
    )

    print(f"[model] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
    print("[model] Ready")
    return model, processor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. LOSS COMPUTATION  (teacher-forced, exact same as training CE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def compute_loss_for_sample(model, processor, user_text: str, gt_text: str,
                             image: Optional[Image.Image],
                             device: str) -> float:
    """
    Teacher-forced cross-entropy loss for one sample.
    Full conversation = user_text + gt_text is tokenised together;
    loss is computed only over the gt_text tokens (assistant turn).
    """
    # Build conversation in chat-template format
    messages = [{"role": "user", "content": []}]

    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": user_text})

    # Apply chat template for the prompt portion
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Full text = prompt + ground truth (teacher-forced)
    full_text = prompt_text + gt_text

    if image is not None:
        inputs = processor(
            text=full_text,
            images=[image],
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=full_text,
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Mask: compute loss only on assistant tokens
    prompt_ids = processor(
        text=prompt_text,
        return_tensors="pt",
    )["input_ids"]
    prompt_len = prompt_ids.shape[1]

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100          # mask prompt tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    outputs = model(**inputs, labels=labels)
    return float(outputs.loss.item())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def generate_response(model, processor, user_text: str,
                      image: Optional[Image.Image],
                      device: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": user_text})

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if image is not None:
        inputs = processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        )
    else:
        inputs = processor(text=prompt_text, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy for deterministic eval
        temperature=1.0,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    # Decode only generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids    = out_ids[:, prompt_len:]
    return processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. WAYMO EVAL LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_waymo(model, processor, records: List[dict],
               image_root: str, device: str,
               max_new_tokens: int, skip_generation: bool) -> Tuple[dict, List[dict]]:
    """
    Returns:
        agg_metrics : dict with aggregate numbers
        sample_rows : list of per-sample result dicts (for CSV)
    """
    image_root = Path(image_root)
    losses, ades, fdes, parse_rates = [], [], [], []
    sample_rows = []

    print(f"\n[waymo eval] {len(records)} samples")
    for i, rec in enumerate(records):
        user_text, gt_text = extract_conversations(rec)

        # Load front-camera image
        img_rel  = rec.get("image", "")
        img_path = image_root / img_rel
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] sample {i}: cannot open {img_path} ({e}). Skipping.")
            continue

        # ── Loss ──────────────────────────────────────────────────────────
        try:
            loss = compute_loss_for_sample(
                model, processor, user_text, gt_text, image, device
            )
        except Exception as e:
            print(f"  [warn] sample {i}: loss failed ({e})")
            loss = float("nan")

        ppl = math.exp(loss) if not math.isnan(loss) and loss < 30 else float("nan")
        losses.append(loss)

        # ── Generation + ADE/FDE ──────────────────────────────────────────
        ade, fde, parsed, gen_text = float("nan"), float("nan"), 0, ""

        if not skip_generation:
            try:
                gen_text = generate_response(
                    model, processor, user_text, image, device, max_new_tokens
                )
                gt_pts   = parse_waypoints(gt_text)
                pred_pts = parse_waypoints(gen_text)

                if gt_pts is not None and pred_pts is not None:
                    ade, fde = compute_ade_fde(pred_pts, gt_pts)
                    parsed   = 1
                    ades.append(ade)
                    fdes.append(fde)

            except Exception as e:
                print(f"  [warn] sample {i}: generation failed ({e})")

        parse_rates.append(parsed)

        row = {
            "sample_idx":    i,
            "image":         img_rel,
            "loss":          round(loss, 4),
            "perplexity":    round(ppl, 4) if not math.isnan(ppl) else "nan",
            "ade_m":         round(ade, 4) if not math.isnan(ade) else "nan",
            "fde_m":         round(fde, 4) if not math.isnan(fde) else "nan",
            "waypoint_parsed": parsed,
            "gt_text":       gt_text[:300].replace("\n", " "),
            "gen_text":      gen_text[:300].replace("\n", " "),
        }
        sample_rows.append(row)

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            mean_loss = float(np.nanmean(losses)) if losses else float("nan")
            mean_ade  = float(np.nanmean(ades))   if ades  else float("nan")
            print(f"  [{i+1}/{len(records)}] "
                  f"loss={mean_loss:.4f}  "
                  f"ADE={mean_ade:.3f}m  "
                  f"parse_rate={np.mean(parse_rates)*100:.1f}%")

    # ── Aggregates ────────────────────────────────────────────────────────
    valid_losses = [l for l in losses if not math.isnan(l)]
    mean_loss    = float(np.mean(valid_losses)) if valid_losses else float("nan")
    mean_ppl     = math.exp(mean_loss) if not math.isnan(mean_loss) and mean_loss < 30 \
                   else float("nan")

    agg = {
        "n_samples":     len(sample_rows),
        "mean_loss":     round(mean_loss, 4),
        "perplexity":    round(mean_ppl, 4) if not math.isnan(mean_ppl) else "nan",
        "mean_ade_m":    round(float(np.nanmean(ades)), 4) if ades else "nan",
        "mean_fde_m":    round(float(np.nanmean(fdes)), 4) if fdes else "nan",
        "parse_rate_pct":round(float(np.mean(parse_rates)) * 100, 1) if parse_rates else "nan",
        "n_parsed":      sum(parse_rates),
    }
    return agg, sample_rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. REASONING EVAL LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_reasoning(model, processor, records: List[dict],
                   device: str, max_new_tokens: int,
                   skip_generation: bool) -> Tuple[dict, List[dict]]:
    losses, rouge_ls = [], []
    sample_rows = []

    print(f"\n[reasoning eval] {len(records)} samples")
    for i, rec in enumerate(records):
        user_text, gt_text = extract_conversations(rec)

        # ── Loss ──────────────────────────────────────────────────────────
        try:
            loss = compute_loss_for_sample(
                model, processor, user_text, gt_text, image=None, device=device
            )
        except Exception as e:
            print(f"  [warn] sample {i}: loss failed ({e})")
            loss = float("nan")

        ppl = math.exp(loss) if not math.isnan(loss) and loss < 30 else float("nan")
        losses.append(loss)

        # ── Generation + ROUGE-L ──────────────────────────────────────────
        rouge_l, gen_text = float("nan"), ""

        if not skip_generation:
            try:
                gen_text = generate_response(
                    model, processor, user_text, image=None,
                    device=device, max_new_tokens=max_new_tokens
                )
                rouge_l = compute_rouge_l(gen_text, gt_text)
                rouge_ls.append(rouge_l)
            except Exception as e:
                print(f"  [warn] sample {i}: generation failed ({e})")

        row = {
            "sample_idx":  i,
            "loss":        round(loss, 4),
            "perplexity":  round(ppl, 4) if not math.isnan(ppl) else "nan",
            "rouge_l":     round(rouge_l, 4) if not math.isnan(rouge_l) else "nan",
            "gt_text":     gt_text[:300].replace("\n", " "),
            "gen_text":    gen_text[:300].replace("\n", " "),
        }
        sample_rows.append(row)

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            mean_loss   = float(np.nanmean(losses)) if losses else float("nan")
            mean_rouge  = float(np.nanmean(rouge_ls)) if rouge_ls else float("nan")
            print(f"  [{i+1}/{len(records)}] "
                  f"loss={mean_loss:.4f}  "
                  f"ROUGE-L={mean_rouge:.4f}")

    valid_losses = [l for l in losses if not math.isnan(l)]
    mean_loss    = float(np.mean(valid_losses)) if valid_losses else float("nan")
    mean_ppl     = math.exp(mean_loss) if not math.isnan(mean_loss) and mean_loss < 30 \
                   else float("nan")

    agg = {
        "n_samples":      len(sample_rows),
        "mean_loss":      round(mean_loss, 4),
        "perplexity":     round(mean_ppl, 4) if not math.isnan(mean_ppl) else "nan",
        "mean_rouge_l":   round(float(np.nanmean(rouge_ls)), 4) if rouge_ls else "nan",
    }
    return agg, sample_rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. CSV WRITER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [saved] {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load val data ──────────────────────────────────────────────────────
    print("[data] Loading JSONL files...")

    if args.waymo_val_jsonl:
        waymo_val = load_jsonl(args.waymo_val_jsonl)
        print(f"  Waymo val: {len(waymo_val)} records (from dedicated val split)")
    else:
        waymo_all = load_jsonl(args.waymo_jsonl)
        waymo_val = sample_val(waymo_all, args.val_fraction,
                               args.val_seed, args.max_val_waymo)
        print(f"  Waymo val: {len(waymo_val)} records "
              f"(sampled {args.val_fraction*100:.0f}% of {len(waymo_all)})")

    if args.reason_val_jsonl:
        reason_val = load_jsonl(args.reason_val_jsonl)
        print(f"  Reasoning val: {len(reason_val)} records (from dedicated val split)")
    else:
        reason_all = load_jsonl(args.reason_jsonl)
        reason_val = sample_val(reason_all, args.val_fraction,
                                args.val_seed, args.max_val_reason)
        print(f"  Reasoning val: {len(reason_val)} records "
              f"(sampled {args.val_fraction*100:.0f}% of {len(reason_all)})")

    # ── Loop over adapters ─────────────────────────────────────────────────
    all_results = {}

    for adapter_path in args.adapter_path:
        ckpt_name = Path(adapter_path).name   # e.g. "checkpoint-200" or "final"
        print(f"\n{'='*70}")
        print(f" Evaluating: {ckpt_name}")
        print(f"{'='*70}")

        t0 = time.time()

        # Load model fresh for each adapter
        model, processor = load_model_and_processor(
            args.base_model, adapter_path, args.device, args.dtype
        )

        # ── Waymo ────────────────────────────────────────────────────────
        waymo_agg, waymo_rows = eval_waymo(
            model, processor, waymo_val,
            image_root=args.image_root,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            skip_generation=args.skip_generation,
        )
        write_csv(waymo_rows,
                  str(out_dir / f"samples_waymo_{ckpt_name}.csv"))

        # ── Reasoning ────────────────────────────────────────────────────
        reason_agg, reason_rows = eval_reasoning(
            model, processor, reason_val,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            skip_generation=args.skip_generation,
        )
        write_csv(reason_rows,
                  str(out_dir / f"samples_reasoning_{ckpt_name}.csv"))

        elapsed = time.time() - t0

        all_results[ckpt_name] = {
            "adapter_path": adapter_path,
            "elapsed_sec":  round(elapsed, 1),
            "waymo":        waymo_agg,
            "reasoning":    reason_agg,
        }

        # ── Print summary ────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  {ckpt_name}  ({elapsed/60:.1f} min)")
        print(f"  Waymo  │ loss={waymo_agg['mean_loss']}  "
              f"ppl={waymo_agg['perplexity']}  "
              f"ADE={waymo_agg['mean_ade_m']}m  "
              f"FDE={waymo_agg['mean_fde_m']}m  "
              f"parse={waymo_agg['parse_rate_pct']}%")
        print(f"  Reason │ loss={reason_agg['mean_loss']}  "
              f"ppl={reason_agg['perplexity']}  "
              f"ROUGE-L={reason_agg['mean_rouge_l']}")
        print(f"{'─'*50}")

        # Free VRAM before next adapter
        del model
        torch.cuda.empty_cache()

    # ── Save aggregate JSON ────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    # Append to existing if present (useful for incremental runs)
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[done] Metrics saved to {metrics_path}")

    # ── Comparison table if multiple adapters ─────────────────────────────
    if len(args.adapter_path) > 1:
        print(f"\n{'─'*70}")
        print(f"  {'Checkpoint':<20} {'W-loss':>8} {'W-ADE':>8} "
              f"{'W-FDE':>8} {'R-loss':>8} {'ROUGE-L':>8}")
        print(f"{'─'*70}")
        for ck, res in all_results.items():
            w = res["waymo"]
            r = res["reasoning"]
            print(f"  {ck:<20} "
                  f"{str(w['mean_loss']):>8} "
                  f"{str(w['mean_ade_m']):>8} "
                  f"{str(w['mean_fde_m']):>8} "
                  f"{str(r['mean_loss']):>8} "
                  f"{str(r['mean_rouge_l']):>8}")
        print(f"{'─'*70}")


if __name__ == "__main__":
    main()