#!/usr/bin/env python3
"""
save_results.py — Generate visual result cards from tinyvlm batch inference.

Reads outputs/batch_results.json and the corresponding images from images/,
then generates:
  - Per-image result cards (image + model response overlay)
  - A stitched report.png grid of all cards
  - A results.json summary

Usage:
    python save_results.py [--output_dir DIR] [--input JSON] [--images_dir DIR]
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ─── Configuration ───────────────────────────────────────────
CARD_WIDTH = 900
IMAGE_HEIGHT = 400
PADDING = 20
TEXT_COLOR = (240, 240, 240)
BG_COLOR = (30, 30, 35)
HEADER_COLOR = (50, 120, 220)
ACCENT_COLOR = (80, 200, 120)
STAT_COLOR = (180, 180, 190)
GRID_COLS = 4


def get_font(size=14):
    """Try to load a monospace/sans font, fallback to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def get_bold_font(size=14):
    """Try to load a bold font."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return get_font(size)


def wrap_text(text, font, max_width, draw):
    """Word-wrap text to fit within max_width pixels."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for w in words[1:]:
            test = current + " " + w
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                lines.append(current)
                current = w
        lines.append(current)
    return lines


def create_card(image_path, result, fonts):
    """Create a single result card with image + analysis text."""
    font, font_bold, font_small, font_title = fonts

    # Load and resize the source image
    try:
        img = Image.open(image_path).convert("RGB")
        # Fit to card width while maintaining aspect ratio
        ratio = CARD_WIDTH / img.width
        new_h = min(int(img.height * ratio), IMAGE_HEIGHT)
        img = img.resize((CARD_WIDTH, new_h), Image.LANCZOS)
    except Exception:
        img = Image.new("RGB", (CARD_WIDTH, IMAGE_HEIGHT), (60, 60, 60))
        d = ImageDraw.Draw(img)
        d.text((CARD_WIDTH // 2 - 80, IMAGE_HEIGHT // 2), "Image not found", fill=(200, 60, 60), font=font)
        new_h = IMAGE_HEIGHT

    # Create temporary draw for text measurement
    tmp = Image.new("RGB", (CARD_WIDTH, 2000), BG_COLOR)
    tmp_draw = ImageDraw.Draw(tmp)

    # Prepare response text
    response = result.get("model_response", "No response")
    wrapped = wrap_text(response, font, CARD_WIDTH - 2 * PADDING, tmp_draw)

    # Calculate text height
    line_height = font.size + 4 if hasattr(font, 'size') else 18
    text_block_height = len(wrapped) * line_height

    # Stats bar height
    stats_height = 50

    # Total card height
    header_height = 45
    total_height = header_height + new_h + PADDING + stats_height + PADDING + text_block_height + PADDING * 2

    # Create card
    card = Image.new("RGB", (CARD_WIDTH, total_height), BG_COLOR)
    draw = ImageDraw.Draw(card)

    y = 0

    # Header bar
    draw.rectangle([0, 0, CARD_WIDTH, header_height], fill=HEADER_COLOR)
    name = result.get("image", "unknown")
    draw.text((PADDING, 10), f"  {name}", fill=(255, 255, 255), font=font_title)
    y += header_height

    # Paste image
    card.paste(img, (0, y))
    y += new_h

    # Stats bar
    y += 8
    tok = result.get("tokens_generated", 0)
    tps = result.get("tokens_per_sec", 0)
    prefill = result.get("prefill_ms", 0)
    gen_ms = result.get("generate_ms", 0)
    total_ms = prefill + gen_ms

    stats_text = f"{tok} tokens  |  {tps:.1f} tok/s  |  prefill {prefill:.0f}ms  |  gen {gen_ms:.0f}ms  |  total {total_ms/1000:.1f}s"
    draw.rectangle([PADDING, y, CARD_WIDTH - PADDING, y + 30], fill=(40, 45, 50), outline=(60, 65, 70))
    draw.text((PADDING + 10, y + 6), stats_text, fill=ACCENT_COLOR, font=font_small)
    y += 40

    # Response text
    y += 5
    for line in wrapped:
        draw.text((PADDING, y), line, fill=TEXT_COLOR, font=font)
        y += line_height

    return card


def create_report_grid(cards, output_path):
    """Stitch all cards into a grid image."""
    if not cards:
        return

    cols = min(GRID_COLS, len(cards))
    rows = (len(cards) + cols - 1) // cols

    # Scale cards down for the grid
    thumb_width = 450
    thumbs = []
    for c in cards:
        ratio = thumb_width / c.width
        th = int(c.height * ratio)
        thumbs.append(c.resize((thumb_width, th), Image.LANCZOS))

    max_h = max(t.height for t in thumbs)
    gap = 6

    grid_w = cols * thumb_width + (cols + 1) * gap
    grid_h = rows * max_h + (rows + 1) * gap + 60  # extra for title

    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 25))
    draw = ImageDraw.Draw(grid)

    # Title
    title_font = get_bold_font(24)
    draw.text((gap + 10, 15), "SafetyVLM — Inference Results (Qwen2.5-VL-7B Q4_K_M)", fill=(255, 255, 255), font=title_font)

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = gap + c * (thumb_width + gap)
        y = 60 + gap + r * (max_h + gap)
        grid.paste(thumb, (x, y))

    grid.save(output_path, quality=92)
    print(f"  Report grid saved: {output_path} ({grid_w}x{grid_h})")


def main():
    parser = argparse.ArgumentParser(description="Generate visual result cards from batch inference")
    parser.add_argument("--output_dir", default="outputs/results", help="Output directory")
    parser.add_argument("--input", default="outputs/batch_results.json", help="Input JSON from tinyvlm")
    parser.add_argument("--images_dir", default="images", help="Directory with source images")
    args = parser.parse_args()

    # Load results
    with open(args.input) as f:
        results = json.load(f)

    print(f"[save_results] Loaded {len(results)} results from {args.input}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load fonts
    fonts = (get_font(14), get_bold_font(14), get_font(12), get_bold_font(18))

    cards = []
    for i, result in enumerate(results):
        image_name = result.get("image", f"image_{i:02d}.png")
        image_path = os.path.join(args.images_dir, image_name)

        if not os.path.exists(image_path):
            # Try image_path from result
            alt = result.get("image_path", "")
            if os.path.exists(alt):
                image_path = alt

        card = create_card(image_path, result, fonts)
        cards.append(card)

        # Save individual card
        card_name = f"card_{i+1:02d}_{Path(image_name).stem}.png"
        card_path = os.path.join(args.output_dir, card_name)
        card.save(card_path, quality=90)
        print(f"  [{i+1:2d}/{len(results)}] {card_name} ({card.width}x{card.height})")

    # Save report grid
    report_path = os.path.join(args.output_dir, "report.png")
    create_report_grid(cards, report_path)

    # Save results JSON copy
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {results_json_path}")

    print(f"\n[save_results] Done! {len(cards)} cards + report saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
