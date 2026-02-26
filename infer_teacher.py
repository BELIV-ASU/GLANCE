"""
infer_checkpoint400.py  –  Inference with SafetyVLM Teacher (checkpoint-400)
on val data (MCQ + QA) from driving_road_data + handbook images.

Uses:
  - Base model: Qwen/Qwen2.5-VL-32B-Instruct
  - LoRA adapter: checkpoints/checkpoint-400
  - Val data: driving_road_data/val/question_mcq_val.json
              driving_road_data/val/question_qa_val.json
  - Handbook images: driving_handbook_data/img/

Usage:
  python infer_checkpoint400.py
  python infer_checkpoint400.py --num_samples 20
  python infer_checkpoint400.py --data_type mcq --num_samples 5
  python infer_checkpoint400.py --data_type handbook --num_samples 10
"""

import os, sys, json, random, argparse, torch
from pathlib import Path

# Force cublasLt backend (same workaround as training)
torch.backends.cuda.preferred_blas_library("cublaslt")

from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as ModelClass

# ---------------------------------------------------------------------------
#  Monkey-patch: fix CUBLAS_STATUS_INVALID_VALUE in rotary embeddings
# ---------------------------------------------------------------------------
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25vl

_orig_rope_forward = _qwen25vl.Qwen2_5_VLRotaryEmbedding.forward

def _patched_rope_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        freqs = torch.einsum("abcd,abde->abce", inv_freq_expanded, position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

_qwen25vl.Qwen2_5_VLRotaryEmbedding.forward = _patched_rope_forward


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_MODEL   = "Qwen/Qwen2.5-VL-32B-Instruct"
DEFAULT_ADAPTER      = str(SCRIPT_DIR / "checkpoints" / "checkpoint-400")
DEFAULT_VAL_DIR      = "/scratch/jnolas77/driving_road_data/val"
DEFAULT_IMG_ROOT     = "/scratch/jnolas77/driving_road_data"
DEFAULT_HANDBOOK_DIR = "/scratch/jnolas77/driving_handbook_data"

SYSTEM_PROMPT = (
    "You are SafetyVLM-Teacher, an expert international driving instructor and "
    "traffic-rule analyst.  You have encyclopaedic knowledge of driving regulations "
    "from every country, in multiple languages.  When shown a traffic sign, road "
    "scenario, or handbook excerpt you:\n"
    "1. Identify the applicable rule(s) and jurisdiction.\n"
    "2. Explain the reasoning behind the rule.\n"
    "3. Describe correct driver behaviour, including edge cases.\n"
    "4. If the question is exam-style, give the correct answer with a clear explanation.\n"
    "Always be precise, cite the country/region, and respond in the language of the query "
    "unless asked otherwise."
)


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
def load_model(args):
    """Load Qwen2.5-VL-32B + checkpoint-400 LoRA adapter."""
    print(f"[*] Loading base model: {args.base_model}")
    print(f"[*] LoRA adapter: {args.adapter_path}")

    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = ModelClass.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    )

    if os.path.isdir(args.adapter_path):
        print(f"[*] Loading LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"[!] WARNING: adapter path not found: {args.adapter_path}")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )

    return model, processor


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
def load_val_data(val_dir, data_type="both"):
    """Load val JSON data (mcq, qa, or both)."""
    samples = []

    if data_type in ("mcq", "both"):
        mcq_path = os.path.join(val_dir, "question_mcq_val.json")
        if os.path.isfile(mcq_path):
            with open(mcq_path, "r") as f:
                mcq_data = json.load(f)
            print(f"[*] Loaded {len(mcq_data)} MCQ samples from {mcq_path}")
            samples.extend(mcq_data)
        else:
            print(f"[!] MCQ file not found: {mcq_path}")

    if data_type in ("qa", "both"):
        qa_path = os.path.join(val_dir, "question_qa_val.json")
        if os.path.isfile(qa_path):
            with open(qa_path, "r") as f:
                qa_data = json.load(f)
            print(f"[*] Loaded {len(qa_data)} QA samples from {qa_path}")
            samples.extend(qa_data)
        else:
            print(f"[!] QA file not found: {qa_path}")

    return samples


def format_mcq_prompt(sample):
    """Format an MCQ sample into a prompt string."""
    prompt = f"{sample['question']}\n\n"
    for key in ["optionA", "optionB", "optionC", "optionD", "optionE", "optionF"]:
        if key in sample and sample[key]:
            letter = key.replace("option", "")
            prompt += f"  {letter}. {sample[key]}\n"
    prompt += "\nSelect the correct answer and explain your reasoning."
    return prompt


def format_qa_prompt(sample):
    """Format a QA sample into a prompt string."""
    return sample["question"]


def load_handbook_data(handbook_dir, num_samples=10, seed=42):
    """Load handbook entries that have images actually on disk."""
    data_json = os.path.join(handbook_dir, "data.json")
    if not os.path.isfile(data_json):
        print(f"[!] Handbook data.json not found: {data_json}")
        return []

    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect entries with images that exist on disk
    entries_with_images = []
    for entry in data:
        if not entry.get("imgs"):
            continue
        for img_info in entry["imgs"]:
            rel = img_info["img_path"].replace("\\", "/")
            abs_path = os.path.join(handbook_dir, rel)
            if os.path.isfile(abs_path):
                entries_with_images.append({
                    "tag": "handbook",
                    "country": entry.get("country", "Unknown"),
                    "vehicle_type": entry.get("vehicle_type", "general"),
                    "language": entry.get("language", "English"),
                    "title": entry.get("title", ""),
                    "text": entry.get("text", ""),
                    "img_abs_path": abs_path,
                    "img_rel_path": rel,
                    "caption": img_info.get("figure_caption", []),
                })

    print(f"[*] Found {len(entries_with_images)} handbook entries with images on disk")

    # Sample a subset
    random.seed(seed)
    n = min(num_samples, len(entries_with_images))
    selected = random.sample(entries_with_images, n)
    return selected


def format_handbook_prompt(sample):
    """Format a handbook sample into a prompt for vision inference."""
    country = sample["country"]
    vtype = sample["vehicle_type"]
    title = sample["title"]
    text = sample["text"][:300].strip()

    prompt = (
        f"This image is from the {country} driving handbook ({vtype}).\n"
        f"Section: \"{title}\"\n\n"
    )
    if text:
        prompt += f"Context: {text}\n\n"
    prompt += (
        "Analyse the image step-by-step:\n"
        "1. Identify the traffic element shown.\n"
        "2. Recall the applicable rule and jurisdiction.\n"
        "3. Describe the correct driver response.\n"
        "4. Note any edge cases or common mistakes."
    )
    return prompt


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------
def generate(model, processor, prompt, image_path=None,
             max_new_tokens=512, temperature=0.7):
    """Generate a response, optionally with an image."""
    user_content = []

    # Try to load image if path is provided
    if image_path and os.path.isfile(image_path):
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            user_content.append({"type": "image", "image": img})
            print(f"    [image loaded: {image_path}]")
        except Exception as e:
            print(f"    [image load failed: {e}]")

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def run_inference(args):
    """Main inference loop."""
    # Load model
    model, processor = load_model(args)

    # --------------- Collect samples ---------------
    selected = []

    if args.data_type in ("mcq", "qa", "both"):
        # Load val road data
        all_val = load_val_data(args.val_dir, args.data_type)
        if all_val:
            random.seed(args.seed)
            n_val = min(args.num_samples, len(all_val))
            selected.extend(random.sample(all_val, n_val))

    if args.data_type in ("handbook", "both"):
        # Load handbook entries with real images
        n_handbook = args.num_handbook if args.data_type == "handbook" else max(5, args.num_samples // 2)
        handbook_samples = load_handbook_data(args.handbook_dir, num_samples=n_handbook, seed=args.seed)
        selected.extend(handbook_samples)

    if not selected:
        print("[!] No data found. Exiting.")
        return

    num = len(selected)
    print(f"\n[*] Total samples for inference: {num}\n")

    results = []
    for i, sample in enumerate(selected, 1):
        tag = sample.get("tag", "unknown")
        country = sample.get("country", ["Unknown"])
        if isinstance(country, list):
            country = country[0] if country else "Unknown"
        sign_desc = sample.get("sign", [""])
        if isinstance(sign_desc, list):
            sign_desc = sign_desc[0] if sign_desc else ""

        # ---- Handbook samples (have real images) ----
        if tag == "handbook":
            prompt = format_handbook_prompt(sample)
            gt_answer_text = f"[{country}] {sample.get('title', 'N/A')}"
            img_abs = sample.get("img_abs_path")
            img_rel = sample.get("img_rel_path", "")
            has_image = img_abs and os.path.isfile(img_abs)
        # ---- MCQ samples ----
        elif tag == "mcq":
            prompt = format_mcq_prompt(sample)
            gt_answer_key = sample.get("answer", [None])
            if isinstance(gt_answer_key, list):
                gt_answer_key = gt_answer_key[0] if gt_answer_key else None
            gt_answer_text = sample.get(gt_answer_key, gt_answer_key) if gt_answer_key else "N/A"
            img_rel = sample.get("img_path", "")
            img_abs = os.path.join(args.img_root, img_rel) if img_rel else None
            has_image = img_abs and os.path.isfile(img_abs)
        # ---- QA samples ----
        else:
            prompt = format_qa_prompt(sample)
            gt_answer = sample.get("answer", [])
            gt_answer_text = gt_answer[0] if isinstance(gt_answer, list) and gt_answer else str(gt_answer)
            img_rel = sample.get("img_path", "")
            img_abs = os.path.join(args.img_root, img_rel) if img_rel else None
            has_image = img_abs and os.path.isfile(img_abs)

        print(f"{'='*70}")
        print(f"  Sample {i}/{num}  |  Type: {tag}  |  Country: {country}")
        if tag == "handbook":
            print(f"  Title: {sample.get('title', '')[:80]}")
        else:
            print(f"  Sign: {sign_desc[:80]}...")
        print(f"  Image: {img_rel}")
        print(f"  Image available: {'YES' if has_image else 'No (text-only)'}")
        print(f"{'='*70}")
        print(f"\n  PROMPT:\n{prompt[:400]}{'...' if len(prompt)>400 else ''}\n")

        # Generate
        response = generate(
            model, processor, prompt,
            image_path=img_abs if has_image else None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"  MODEL RESPONSE:\n{response}\n")
        print(f"  GROUND TRUTH:\n{gt_answer_text}\n")

        results.append({
            "sample_idx": i,
            "tag": tag,
            "country": country,
            "sign": sign_desc if tag != "handbook" else sample.get("title", ""),
            "prompt": prompt,
            "image_path": img_rel,
            "image_available": has_image,
            "model_response": response,
            "ground_truth": gt_answer_text,
        })

    # Save results
    out_path = os.path.join(SCRIPT_DIR, "inference_results_ckpt400.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"  Model: {args.base_model}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Samples processed: {len(results)}")
    mcq_count  = sum(1 for r in results if r["tag"] == "mcq")
    qa_count   = sum(1 for r in results if r["tag"] == "qa")
    hb_count   = sum(1 for r in results if r["tag"] == "handbook")
    img_count  = sum(1 for r in results if r["image_available"])
    print(f"  MCQ: {mcq_count}  |  QA: {qa_count}  |  Handbook (with image): {hb_count}")
    print(f"  With image: {img_count}  |  Text-only: {len(results) - img_count}")

    # Quick accuracy check for MCQ
    if mcq_count > 0:
        correct = 0
        for r in results:
            if r["tag"] != "mcq":
                continue
            # Check if the ground truth option text appears in the response
            gt = r["ground_truth"].lower().strip()
            resp = r["model_response"].lower()
            if gt in resp:
                correct += 1
        print(f"  MCQ rough accuracy: {correct}/{mcq_count} ({100*correct/mcq_count:.1f}%)")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference with checkpoint-400 on val data")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--val_dir", default=DEFAULT_VAL_DIR)
    parser.add_argument("--img_root", default=DEFAULT_IMG_ROOT)
    parser.add_argument("--handbook_dir", default=DEFAULT_HANDBOOK_DIR,
                        help="Path to driving_handbook_data with data.json and img/")
    parser.add_argument("--data_type", choices=["mcq", "qa", "both", "handbook"], default="both",
                        help="Which data to use: mcq, qa, both (val + handbook), or handbook only")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of val samples to run inference on")
    parser.add_argument("--num_handbook", type=int, default=10,
                        help="Number of handbook image samples (used with --data_type handbook)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
