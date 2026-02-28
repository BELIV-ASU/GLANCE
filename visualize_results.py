"""
visualize_results.py – Render each inference result as an image:
  left side = camera frame, right side = model output + ground truth text.

Usage:
  python visualize_results.py
  python visualize_results.py --results inference_results_ckpt200.json --out_dir viz_ckpt200
"""

import os, json, argparse, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent


def get_font(size=18):
    """Try to load a monospace TTF font; fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text, width=60):
    """Wrap text to a fixed character width."""
    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        lines.extend(wrapped)
    return lines


def render_result(result, out_path, img_height=720, font_size=16, text_width_chars=70):
    """Render a single result: image on left, text on right."""
    font = get_font(font_size)
    line_height = font_size + 4

    # Load camera image
    img_path = result.get("image_path", "")
    if img_path and os.path.isfile(img_path):
        cam_img = Image.open(img_path).convert("RGB")
    else:
        # Grey placeholder
        cam_img = Image.new("RGB", (1600, 900), (80, 80, 80))
        draw = ImageDraw.Draw(cam_img)
        draw.text((400, 400), "No image available", fill=(200, 200, 200), font=get_font(30))

    # Scale camera image to target height
    aspect = cam_img.width / cam_img.height
    img_w = int(img_height * aspect)
    cam_img = cam_img.resize((img_w, img_height), Image.LANCZOS)

    # Build text block
    idx = result.get("sample_idx", "?")
    prompt = result.get("prompt", "")
    response = result.get("model_response", "")
    gt = result.get("ground_truth", "")

    text_lines = []
    text_lines.append(f"--- Sample {idx} ---")
    text_lines.append("")
    text_lines.append("PROMPT:")
    text_lines.extend(wrap_text(prompt[:500], text_width_chars))
    text_lines.append("")
    text_lines.append("MODEL RESPONSE:")
    text_lines.extend(wrap_text(response, text_width_chars))
    text_lines.append("")
    text_lines.append("GROUND TRUTH:")
    text_lines.extend(wrap_text(gt, text_width_chars))

    # Compute text panel size
    text_panel_w = int(text_width_chars * font_size * 0.62)  # approx char width
    text_panel_h = max(img_height, len(text_lines) * line_height + 40)

    # Create canvas
    canvas_w = img_w + text_panel_w + 20  # 20px gap
    canvas_h = max(img_height, text_panel_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    # Paste camera image (vertically centered)
    y_offset = (canvas_h - img_height) // 2
    canvas.paste(cam_img, (0, y_offset))

    # Draw text
    draw = ImageDraw.Draw(canvas)
    x_text = img_w + 15
    y = 10
    for line in text_lines:
        if line.startswith("---") or line.startswith("PROMPT:") or line.startswith("MODEL RESPONSE:") or line.startswith("GROUND TRUTH:"):
            color = (100, 200, 255)  # cyan for headers
        elif line.startswith("GROUND TRUTH:"):
            color = (100, 255, 100)  # green
        else:
            color = (220, 220, 220)  # light grey
        draw.text((x_text, y), line, fill=color, font=font)
        y += line_height
        if y > canvas_h - 20:
            break

    canvas.save(out_path, quality=92)


def main():
    parser = argparse.ArgumentParser(description="Visualize inference results as images")
    parser.add_argument("--results", default=str(SCRIPT_DIR / "inference_results_ckpt200.json"),
                        help="Path to inference results JSON")
    parser.add_argument("--out_dir", default=str(SCRIPT_DIR / "viz_ckpt200"),
                        help="Output directory for rendered images")
    parser.add_argument("--img_height", type=int, default=720,
                        help="Height of the camera image panel")
    parser.add_argument("--font_size", type=int, default=16)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[*] Rendering {len(results)} results to {args.out_dir}/")
    for r in results:
        idx = r.get("sample_idx", 0)
        out_path = os.path.join(args.out_dir, f"sample_{idx:03d}.png")
        render_result(r, out_path, img_height=args.img_height, font_size=args.font_size)
        print(f"  [{idx}/{len(results)}] → {out_path}")

    print(f"\n[*] Done. {len(results)} images saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
