"""
infer_checkpoint200.py  –  Inference with DriveLM Teacher (checkpoint-200)
on DriveLM val data (JSONL with scene-based VQA + nuscenes camera images).

Uses:
  - Base model: Qwen/Qwen2.5-VL-32B-Instruct
  - LoRA adapter: checkpoints/checkpoint-200
  - Val data: data_drivelm/val.jsonl  (built by data.py from DriveLM)

Usage:
  python infer_checkpoint200.py
  python infer_checkpoint200.py --num_samples 20
  python infer_checkpoint200.py --max_new_tokens 256
"""

import os, sys, json, random, argparse, torch
import numpy as np
from pathlib import Path
from PIL import Image

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


def _load_image_robust(img_path):
    """Load a PIL image, handling 16-bit depth maps correctly."""
    pil_img = Image.open(img_path)
    if pil_img.mode in ("I;16", "I"):
        arr = np.array(pil_img, dtype=np.float32)
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min > 1e-6:
            arr = (arr - a_min) / (a_max - a_min) * 255.0
        arr = arr.astype(np.uint8)
        pil_img = Image.fromarray(
            np.stack([arr, arr, arr], axis=-1), mode="RGB"
        )
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
DEFAULT_ADAPTER    = str(PROJECT_ROOT / "checkpoints" / "checkpoint-600")
DEFAULT_VAL_JSONL  = str(PROJECT_ROOT / "data_drivelm" / "val.jsonl")


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
def load_model(args):
    """Load Qwen2.5-VL-32B + checkpoint-200 LoRA adapter."""
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
#  Data loading  (DriveLM val.jsonl)
# ---------------------------------------------------------------------------
def load_val_jsonl(val_path, num_samples=10, seed=42):
    """Load samples from DriveLM val.jsonl (chat-message format).

    Each line is: {"messages": [{"role":..., "content":...}, ...]}
    User content may contain {"type":"image","image":"file:///path"} items.
    """
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    print(f"[*] Loaded {len(samples)} val samples from {val_path}")

    random.seed(seed)
    if num_samples > 0 and num_samples < len(samples):
        samples = random.sample(samples, num_samples)

    return samples


def extract_from_messages(messages):
    """Extract the user prompt text, image path, and ground-truth answer
    from DriveLM chat messages."""
    prompt_text = ""
    image_path = None
    gt_answer = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            prompt_text += item.get("text", "")
                        elif item.get("type") == "image":
                            img = item.get("image", "")
                            if img.startswith("file://"):
                                img = img[7:]
                            if os.path.isfile(img):
                                image_path = img
                    elif isinstance(item, str):
                        prompt_text += item
            elif isinstance(content, str):
                prompt_text = content

        elif role == "assistant":
            if isinstance(content, str):
                gt_answer = content
            elif isinstance(content, list):
                gt_answer = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )

    return prompt_text.strip(), image_path, gt_answer.strip()


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------
def generate(model, processor, messages, image_path=None,
             max_new_tokens=512, temperature=0.7):
    """Generate a response from chat messages, optionally with an image."""
    # Build the messages for generation (system + user only, no assistant)
    gen_messages = []
    pil_images = []

    for msg in messages:
        if msg["role"] == "assistant":
            continue  # skip ground truth
        content = msg.get("content", "")
        if msg["role"] == "user" and isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_path = item.get("image", "")
                    if img_path.startswith("file://"):
                        img_path = img_path[7:]
                    if os.path.isfile(img_path):
                        try:
                            pil_img = _load_image_robust(img_path)
                            pil_images.append(pil_img)
                            new_content.append({"type": "image", "image": pil_img})
                            print(f"    [image loaded: {img_path}]")
                        except Exception as e:
                            print(f"    [image load failed: {e}]")
                else:
                    new_content.append(item)
            gen_messages.append({"role": "user", "content": new_content})
        else:
            gen_messages.append(msg)

    text = processor.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=[text], return_tensors="pt", padding=True)
    if pil_images:
        proc_kwargs["images"] = pil_images

    inputs = processor(**proc_kwargs).to(model.device)

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
    """Main inference loop on DriveLM val data."""
    model, processor = load_model(args)

    samples = load_val_jsonl(args.val_jsonl, args.num_samples, args.seed)

    if not samples:
        print("[!] No data found. Exiting.")
        return

    num = len(samples)
    print(f"\n[*] Total samples for inference: {num}\n")

    results = []
    for i, sample in enumerate(samples, 1):
        messages = sample.get("messages", [])
        prompt_text, image_path, gt_answer = extract_from_messages(messages)
        has_image = image_path is not None

        print(f"{'='*70}")
        print(f"  Sample {i}/{num}")
        print(f"  Image: {image_path or 'None (text-only)'}")
        print(f"  Image available: {'YES' if has_image else 'No'}")
        print(f"{'='*70}")
        print(f"\n  PROMPT:\n{prompt_text[:400]}{'...' if len(prompt_text)>400 else ''}\n")

        response = generate(
            model, processor, messages,
            image_path=image_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"  MODEL RESPONSE:\n{response}\n")
        print(f"  GROUND TRUTH:\n{gt_answer}\n")

        results.append({
            "sample_idx": i,
            "prompt": prompt_text[:500],
            "image_path": image_path or "",
            "image_available": has_image,
            "model_response": response,
            "ground_truth": gt_answer,
        })

    # Save results
    results_dir = str(PROJECT_ROOT / "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "inference_results_ckpt200.json")
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
    img_count = sum(1 for r in results if r["image_available"])
    print(f"  With image: {img_count}  |  Text-only: {len(results) - img_count}")

    # Exact-match check
    exact = sum(1 for r in results if r["model_response"].strip() == r["ground_truth"].strip())
    print(f"  Exact match: {exact}/{len(results)} ({100*exact/len(results):.1f}%)")

    # Contains-match check (GT appears in response)
    contains = sum(
        1 for r in results
        if r["ground_truth"].strip().lower() in r["model_response"].strip().lower()
    )
    print(f"  Contains match: {contains}/{len(results)} ({100*contains/len(results):.1f}%)")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference with checkpoint-200 on DriveLM val data")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--val_jsonl", default=DEFAULT_VAL_JSONL,
                        help="Path to DriveLM val.jsonl")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of val samples to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
