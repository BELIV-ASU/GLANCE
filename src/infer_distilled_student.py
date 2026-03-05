"""
infer_distilled_student.py  –  Inference with distilled Qwen2.5-VL-7B student
on DriveLM val data (JSONL with scene-based VQA + nuScenes camera images).

Uses:
  - Base model: Qwen/Qwen2.5-VL-7B-Instruct
  - QLoRA adapter: distilled_student/checkpoints/step-<N>
  - Val data: data_drivelm/val.jsonl

Usage:
  python infer_distilled_student.py
  python infer_distilled_student.py --step 200
  python infer_distilled_student.py --step 400 --num_samples 40 --out_dir my_out/
  python infer_distilled_student.py --no_adapter          # bare 7B baseline
  python infer_distilled_student.py --step 200 --visualize   # infer + render PNGs
"""

import os, sys, json, random, argparse, textwrap, torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Force cublasLt (same workaround as training)
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

_orig_rope = _qwen25vl.Qwen2_5_VLRotaryEmbedding.forward

def _patched_rope(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        freqs = torch.einsum("abcd,abde->abce", inv_freq_expanded,
                              position_ids_expanded).transpose(2, 3)
        emb   = torch.cat((freqs, freqs), dim=-1)
        cos   = emb.cos() * self.attention_scaling
        sin   = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

_qwen25vl.Qwen2_5_VLRotaryEmbedding.forward = _patched_rope


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_DISTIL_DIR   = str(PROJECT_ROOT / "distilled_student")
DEFAULT_VAL_JSONL    = str(PROJECT_ROOT / "data_drivelm" / "val.jsonl")


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------
def _get_font(size: int = 15):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _sanitize(text: str) -> str:
    """Replace curly quotes and other non-latin-1 chars for PIL compat."""
    for src, dst in [('\u2018',"'"),("\u2019","'"),("\u201c",'"'),("\u201d",'"'),
                     ('\u2013','-'),('\u2014','--'),('\u2026','...'),('\u00b0','deg')]:
        text = text.replace(src, dst)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def _wrap(text: str, w: int = 65):
    lines = []
    for para in _sanitize(str(text)).split("\n"):
        lines.extend(textwrap.wrap(para, w) or [""])
    return lines


def render_result(result: dict, val_line: dict, out_path: str,
                  img_h: int = 720, fsz: int = 15):
    """Render one inference result: camera image left, text panel right."""
    font      = _get_font(fsz)
    font_bold = _get_font(fsz + 2)
    lh = fsz + 5

    # -- extract image + prompt from val_line --
    msgs = val_line.get("messages", [])
    img_path, prompt_text = None, ""
    for m in msgs:
        content = m.get("content", [])
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "image":
                    img_path = c.get("image", "").replace("file://", "")
                elif c.get("type") == "text" and m.get("role") == "user":
                    prompt_text = c.get("text", "")

    if img_path and os.path.isfile(img_path):
        cam = _load_image_robust(img_path)
    else:
        cam = Image.new("RGB", (1600, 900), (60, 60, 60))
        ImageDraw.Draw(cam).text((600, 430), "No image",
                                  fill=(180, 180, 180), font=_get_font(28))

    img_w = int(img_h * cam.width / cam.height)
    cam   = cam.resize((img_w, img_h), Image.LANCZOS)

    idx      = result.get("sample_idx", "?")
    response = _sanitize(result.get("model_response", "").strip())
    gt       = _sanitize(result.get("ground_truth", "").strip())
    prompt_s = _sanitize(prompt_text[:600])

    sections = [
        ("header", [f"  Sample {idx}  |  Distilled 7B QLoRA  step-{result.get('step_tag','?')}"]),
        ("div",    [""]),
        ("lp",     ["PROMPT"]),
        ("text",   _wrap(prompt_s, 68)),
        ("div",    [""]),
        ("lr",     ["MODEL RESPONSE"]),
        ("text",   _wrap(response, 68)),
        ("div",    [""]),
        ("lg",     ["GROUND TRUTH"]),
        ("gt",     _wrap(gt, 68)),
    ]

    txt_w    = int(68 * fsz * 0.615)
    n_lines  = sum(len(v) for _, v in sections)
    canvas_h = max(img_h, n_lines * lh + 60)
    canvas   = Image.new("RGB", (img_w + txt_w + 24, canvas_h), (22, 22, 30))
    canvas.paste(cam, (0, (canvas_h - img_h) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.line([(img_w + 6, 0), (img_w + 6, canvas_h)], fill=(60, 60, 80), width=2)

    COLORS = {"header": (255, 215, 0), "lp": (100, 180, 255), "lr": (255, 140, 50),
              "lg": (80, 220, 120), "text": (210, 210, 210), "gt": (150, 255, 150),
              "div": (40, 40, 50)}
    x, y = img_w + 16, 12
    for kind, lines in sections:
        col = COLORS.get(kind, (200, 200, 200))
        f   = font_bold if kind in ("header", "lp", "lr", "lg") else font
        for line in lines:
            if y > canvas_h - 10:
                break
            draw.text((x, y), line, fill=col, font=f)
            y += lh

    canvas.save(out_path, quality=93)


def visualize_results(results: list, val_lines: list, viz_dir: str, step_tag: str):
    """Render all inference results as side-by-side PNG images."""
    os.makedirs(viz_dir, exist_ok=True)
    print(f"\n[viz] Rendering {len(results)} images → {viz_dir}/")
    for r in results:
        idx      = r.get("sample_idx", 0)
        line_idx = idx - 1
        if line_idx < 0 or line_idx >= len(val_lines):
            print(f"  [!] sample_idx={idx} out of val range, skipping viz")
            continue
        r["step_tag"] = step_tag
        out = os.path.join(viz_dir, f"sample_{idx:03d}.png")
        render_result(r, val_lines[line_idx], out)
        print(f"  [{idx:3d}] → {out}")
    print(f"[viz] Done. {len(results)} PNGs saved.\n")


# ---------------------------------------------------------------------------
#  Robust image loader
# ---------------------------------------------------------------------------
def _load_image_robust(img_path: str) -> Image.Image:
    """Load a PIL image; handles 16-bit depth PNGs gracefully."""
    pil = Image.open(img_path)
    if pil.mode in ("I;16", "I"):
        arr = np.array(pil, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = arr.astype(np.uint8)
        pil = Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")
    else:
        pil = pil.convert("RGB")
    return pil


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
def load_model(args):
    adapter_path = None if args.no_adapter else args.adapter_path

    print(f"\n[setup] Base model : {args.base_model}")
    print(f"[setup] Adapter    : {adapter_path or 'none (bare baseline)'}")
    print(f"[setup] Quantise   : 4-bit NF4\n")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = ModelClass.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    )

    if adapter_path:
        if os.path.isdir(adapter_path):
            print(f"[setup] Loading QLoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print(f"[!] WARNING: adapter path not found — {adapter_path}")
            print(f"[!] Running bare base model instead.")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] Params: {total/1e9:.2f}B total  |  {trainable/1e6:.1f}M trainable\n")

    return model, processor


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------
def load_val_jsonl(val_path: str, num_samples: int = 40, seed: int = 42):
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[data] Loaded {len(samples)} val samples from {val_path}")
    random.seed(seed)
    if 0 < num_samples < len(samples):
        samples = random.sample(samples, num_samples)
    return samples


def extract_from_messages(messages):
    """Return (prompt_text, image_path, ground_truth) from chat messages."""
    prompt_text, image_path, gt = "", None, ""
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        prompt_text += str(item)
                        continue
                    if item.get("type") == "text":
                        prompt_text += item.get("text", "")
                    elif item.get("type") == "image":
                        raw = item.get("image", "").replace("file://", "")
                        if os.path.isfile(raw):
                            image_path = raw
            else:
                prompt_text = str(content)
        elif role == "assistant":
            if isinstance(content, str):
                gt = content
            elif isinstance(content, list):
                gt = " ".join(
                    i.get("text","") if isinstance(i,dict) else str(i)
                    for i in content
                )
    return prompt_text.strip(), image_path, gt.strip()


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------
def generate(model, processor, messages, max_new_tokens=256, temperature=0.7):
    gen_msgs   = []
    pil_images = []

    for msg in messages:
        if msg["role"] == "assistant":
            continue
        content = msg.get("content", "")
        if msg["role"] == "user" and isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_path = item.get("image", "").replace("file://", "")
                    if os.path.isfile(img_path):
                        try:
                            pil = _load_image_robust(img_path)
                            pil_images.append(pil)
                            new_content.append({"type": "image", "image": pil})
                        except Exception as e:
                            print(f"    [!] image load failed: {e}")
                    else:
                        print(f"    [!] image missing: {img_path}")
                else:
                    new_content.append(item)
            gen_msgs.append({"role": "user", "content": new_content})
        else:
            gen_msgs.append(msg)

    text = processor.apply_chat_template(
        gen_msgs, tokenize=False, add_generation_prompt=True
    )
    proc_kw = dict(text=[text], return_tensors="pt", padding=True)
    if pil_images:
        proc_kw["images"] = pil_images

    inputs = processor(**proc_kw).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
def run_inference(args):
    model, processor = load_model(args)
    samples          = load_val_jsonl(args.val_jsonl, args.num_samples, args.seed)

    if not samples:
        print("[!] No val samples found. Exiting.")
        return

    print(f"[infer] Running on {len(samples)} samples  |  "
          f"max_new_tokens={args.max_new_tokens}  temperature={args.temperature}\n")

    results = []
    for i, sample in enumerate(samples, 1):
        messages                     = sample.get("messages", [])
        prompt_text, image_path, gt  = extract_from_messages(messages)

        print(f"{'='*70}")
        print(f"  Sample {i}/{len(samples)}")
        print(f"  Image : {image_path or 'none'}")
        print(f"{'='*70}")
        print(f"  PROMPT:\n{prompt_text[:400]}{'...' if len(prompt_text)>400 else ''}\n")

        response = generate(
            model, processor, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"  MODEL RESPONSE:\n{response}\n")
        print(f"  GROUND TRUTH:\n{gt}\n")

        results.append({
            "sample_idx"     : i,
            "prompt"         : prompt_text[:500],
            "image_path"     : image_path or "",
            "image_available": image_path is not None,
            "model_response" : response,
            "ground_truth"   : gt,
        })

    # ── Save JSON ────────────────────────────────────────────────────────────
    results_dir = str(PROJECT_ROOT / "results")
    os.makedirs(results_dir, exist_ok=True)
    tag      = "no_adapter" if args.no_adapter else f"step{args.step}"
    out_json = os.path.join(results_dir, f"distilled_infer_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Results saved → {out_json}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    if args.visualize:
        val_lines = []
        with open(args.val_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    val_lines.append(json.loads(line))
        viz_dir = os.path.join(results_dir, f"viz_{tag}")
        visualize_results(results, val_lines, viz_dir, step_tag=tag)

    # ── Summary ───────────────────────────────────────────────────────────────
    n       = len(results)
    exact   = sum(1 for r in results
                  if r["model_response"].strip() == r["ground_truth"].strip())
    contains= sum(1 for r in results
                  if r["ground_truth"].strip().lower()
                  in r["model_response"].strip().lower())

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Base model   : {args.base_model}")
    adapter_label = "none" if args.no_adapter else args.adapter_path
    print(f"  Adapter      : {adapter_label}")
    print(f"  Samples      : {n}")
    print(f"  With image   : {sum(1 for r in results if r['image_available'])}")
    print(f"  Exact match  : {exact}/{n}  ({100*exact/max(n,1):.1f}%)")
    print(f"  Contains match: {contains}/{n}  ({100*contains/max(n,1):.1f}%)")
    print(f"{'='*70}\n")

    return out_json


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Inference with distilled Qwen2.5-VL-7B student on DriveLM val"
    )
    ap.add_argument("--base_model",     default=DEFAULT_BASE_MODEL,
                    help="HF model ID or local path for the 7B student base")
    ap.add_argument("--distil_dir",     default=DEFAULT_DISTIL_DIR,
                    help="Root of distilled_student/ (contains checkpoints/)")
    ap.add_argument("--step",           type=int, default=200,
                    help="Which distilled checkpoint step to load (e.g. 200, 400)")
    ap.add_argument("--no_adapter",     action="store_true",
                    help="Skip the QLoRA adapter — run bare 7B baseline for comparison")
    ap.add_argument("--val_jsonl",      default=DEFAULT_VAL_JSONL)
    ap.add_argument("--num_samples",    type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--out_dir",        default=DEFAULT_DISTIL_DIR,
                    help="Directory to save inference JSON results")
    ap.add_argument("--visualize",      action="store_true",
                    help="Render results as side-by-side PNG images after inference")
    args = ap.parse_args()

    # Build adapter path from distil_dir + step
    args.adapter_path = os.path.join(args.distil_dir, "checkpoints", f"step-{args.step}")

    run_inference(args)


if __name__ == "__main__":
    main()
