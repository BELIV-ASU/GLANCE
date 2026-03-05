"""
safety_data.py  –  Build instruction-tuning dataset from driving_handbook_data
for Qwen3-VL-32B teacher fine-tuning.

Chain-of-Thought (CoT) + Grounding edition:
  Every assistant answer contains:
    <think>  … step-by-step reasoning …  </think>
    Grounded final answer with [Source: …] citations

Covers five training domains:
  1. Rule reasoning          – explain traffic rules & why they exist
  2. Multilingual knowledge  – preserve original-language entries
  3. Edge cases              – unusual vehicle types, rare conditions
  4. Simulated test Q&A      – exam-style multiple-choice / free-response
  5. Vision grounding        – image-based sign/scenario analysis

Outputs JSONL ready for TRL SFTTrainer.
"""

import json, os, random, re, pathlib
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are SafetyVLM-Teacher, an expert international driving instructor and "
    "traffic-rule analyst.  You have encyclopaedic knowledge of driving regulations "
    "from every country, in multiple languages.\n\n"
    "IMPORTANT – For every response you MUST:\n"
    "1. **Think step-by-step** inside <think>…</think> tags before answering. "
    "Break the problem into: identify jurisdiction → recall applicable rule → "
    "analyse context → consider edge cases → formulate answer.\n"
    "2. **Ground every claim** by citing the source with "
    "[Source: <Country> – <Section Title> – <Vehicle Type>] tags.\n"
    "3. Identify the applicable rule(s) and jurisdiction.\n"
    "4. Explain the reasoning behind the rule.\n"
    "5. Describe correct driver behaviour, including edge cases.\n"
    "6. If the question is exam-style, give the correct answer with a clear explanation.\n\n"
    "Always be precise, cite the country/region, and respond in the language of the query "
    "unless asked otherwise."
)

# ---- User-prompt templates per task type ----

RULE_TEMPLATES = [
    "Based on the following excerpt from the {country} driving handbook ({vehicle_type}), "
    "explain the traffic rule and the reasoning behind it. "
    "Think step-by-step and cite your sources.\n\n"
    "Title: {title}\n\n{text}",

    "You are studying for a {country} driving test ({vehicle_type}). "
    "Read this section and summarise the key rules a driver must follow. "
    "Show your reasoning process.\n\n"
    "**{title}**\n{text}",

    "A driver in {country} encounters this regulation:\n\n\"{text}\"\n\n"
    "Explain what this rule means in practice, any penalties for non-compliance, "
    "and reason through any ambiguities step-by-step.",
]

MULTILINGUAL_TEMPLATES = [
    "The following traffic regulation is from {country} and is written in {language}. "
    "Think through the meaning carefully, then explain it in {language} first, "
    "followed by an English translation. Cite the source.\n\n"
    "**{title}**\n{text}",

    "A {language}-speaking learner driver reads this from the {country} handbook:\n\n"
    "{text}\n\nProvide a detailed chain-of-thought explanation in {language}, "
    "grounded in the handbook text.",
]

EDGE_CASE_TEMPLATES = [
    "Consider this rule from {country} which applies to {vehicle_type} vehicles:\n\n"
    "\"{text}\"\n\n"
    "Think step-by-step through edge cases or unusual scenarios where this rule is "
    "particularly important or where drivers commonly make mistakes. Cite specific parts.",

    "A {vehicle_type} driver in {country} encounters ambiguity in the following regulation:\n\n"
    "**{title}**\n{text}\n\n"
    "Reason through the tricky or uncommon situations a driver should be aware of. "
    "Ground your analysis in the text above.",
]

TEST_TEMPLATES = [
    "**{country} Driving Test – {vehicle_type}**\n\n"
    "Read the following handbook section and generate a challenging exam question "
    "with 4 answer choices (A-D). Show your reasoning for why each incorrect option "
    "is wrong, then provide the correct answer with explanation.\n\n"
    "Section: {title}\n{text}",

    "Create a practical driving-test scenario based on this {country} rule:\n\n"
    "\"{text}\"\n\n"
    "Describe the scenario, reason step-by-step through each option, "
    "identify the correct action, and explain why other actions are wrong. Cite the source.",
]

VISION_TEMPLATES = [
    "This image is from the {country} driving handbook ({vehicle_type}, {language}). "
    "The section title is \"{title}\".\n\n"
    "Analyse the image step-by-step: identify the traffic element → recall the rule → "
    "describe correct driver response → note edge cases. Cite the handbook section.",

    "You see this figure in a {country} {vehicle_type} driving manual:\n\n"
    "Caption: {caption}\n\n"
    "Think through what the image shows, the underlying rule, and how a driver should respond. "
    "Ground your answer in the handbook.",

    "A {language}-speaking learner is studying the {country} driving handbook and encounters "
    "this image under \"{title}\".\n\n"
    "Reason step-by-step about the image content and the relevant traffic rule, "
    "then test the learner with a follow-up question. Cite the source.",
]


# ---------------------------------------------------------------------------
#  Chain-of-Thought answer builders
# ---------------------------------------------------------------------------

def _source_tag(entry: Dict) -> str:
    """Build a grounding citation tag."""
    country = entry.get("country", "Unknown")
    title = (entry.get("title") or "Untitled").strip()
    vtype = entry.get("vehicle_type", "general")
    lang = entry.get("language", "")
    tag = f"[Source: {country} – \"{title}\" – {vtype}"
    if lang and lang != "English":
        tag += f" – {lang}"
    tag += "]"
    return tag


def _build_cot_answer(entry: Dict, task_type: str) -> str:
    """Build a Chain-of-Thought + Grounded answer.

    Structure:
        <think>
        Step 1: Identify jurisdiction & context …
        Step 2: Recall applicable rule …
        Step 3: Analyse the text …
        Step 4: Consider edge cases …
        Step 5: Formulate response …
        </think>

        [Grounded answer with citations]
    """
    country = entry.get("country", "Unknown")
    vtype = entry.get("vehicle_type", "general")
    lang = entry.get("language", "English")
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    src = _source_tag(entry)

    # ---- Build the <think> block ----
    think_steps = []

    # Step 1: Jurisdiction
    think_steps.append(
        f"Step 1 – Identify jurisdiction and context: "
        f"This comes from the {country} driving handbook, specifically the section "
        f"\"{title}\" for {vtype} drivers"
        + (f", written in {lang}" if lang != "English" else "")
        + "."
    )

    # Step 2: Applicable rule
    text_snippet = text[:200].replace("\n", " ").strip()
    if text_snippet:
        think_steps.append(
            f"Step 2 – Recall applicable rule: The handbook states: \"{text_snippet}…\" "
            f"This falls under {country} traffic regulations for {vtype} operators."
        )
    else:
        think_steps.append(
            f"Step 2 – Recall applicable rule: The section \"{title}\" addresses "
            f"traffic regulations in {country} for {vtype} operators."
        )

    # Step 3: Analysis (task-specific)
    if task_type == "rule":
        think_steps.append(
            f"Step 3 – Analyse the rule: This regulation exists to ensure safety for "
            f"{vtype} operators and other road users in {country}. I need to explain "
            f"the specific requirement, the reasoning behind it, and practical implications."
        )
    elif task_type == "multilingual":
        think_steps.append(
            f"Step 3 – Language analysis: The regulation is in {lang}. I must first "
            f"explain it in {lang}, preserving legal nuance, then provide an accurate "
            f"English translation of the key terms."
        )
    elif task_type == "edge_case":
        think_steps.append(
            f"Step 3 – Edge-case analysis: I need to think about unusual scenarios — "
            f"adverse weather, special vehicle categories, intersections with other rules, "
            f"emergency situations, and common misconceptions specific to {vtype} in {country}."
        )
    elif task_type == "test":
        think_steps.append(
            f"Step 3 – Exam analysis: I should create a question that tests deep "
            f"understanding, not just memorisation. The distractors must be plausible "
            f"but clearly wrong upon careful analysis of the {country} regulation."
        )
    elif task_type == "vision":
        think_steps.append(
            f"Step 3 – Visual analysis: I need to describe the visual elements — "
            f"colours, shapes, symbols, road markings — and connect each to the specific "
            f"traffic rule from the {country} handbook."
        )

    # Step 4: Edge cases
    think_steps.append(
        f"Step 4 – Consider edge cases: I should note any exceptions, "
        f"time-of-day restrictions, weather conditions, or vehicle-specific rules "
        f"that modify how this regulation applies for {vtype} in {country}."
    )

    # Step 5: Formulate
    think_steps.append(
        f"Step 5 – Formulate grounded response: I will cite the handbook section "
        f"\"{title}\" and quote relevant text to support each claim."
    )

    think_block = "<think>\n" + "\n".join(think_steps) + "\n</think>\n\n"

    # ---- Build the grounded answer ----
    answer_parts = []

    if task_type == "rule":
        answer_parts.append(f"**Rule Analysis – {country} ({vtype})**\n\n")
        if title:
            answer_parts.append(f"**Section:** {title}\n\n")
        answer_parts.append(f"**Applicable Rule:**\n")
        if text:
            quote = text[:500].strip()
            answer_parts.append(f"> {quote}\n\n")
            answer_parts.append(f"{src}\n\n")
        answer_parts.append(
            f"**Reasoning:** This rule is designed to protect road safety in {country}. "
            f"For {vtype} operators, compliance prevents collisions, protects vulnerable "
            f"road users (pedestrians, cyclists), and avoids legal penalties including "
            f"fines, demerit points, or licence suspension.\n\n"
        )
        answer_parts.append(
            f"**Correct Driver Behaviour:** Follow the regulation as stated in the "
            f"\"{title}\" section of the {country} handbook. When encountering ambiguity, "
            f"adopt the most cautious interpretation. Be aware of local variations and "
            f"recent updates to the highway code.\n\n"
        )
        answer_parts.append(
            f"**Edge Cases to Consider:**\n"
            f"- Emergency vehicles may override standard right-of-way\n"
            f"- Adverse weather (rain, fog, ice) may require extra caution beyond the minimum\n"
            f"- Construction zones may temporarily alter posted rules\n"
            f"- Different rules may apply at night vs. daytime\n\n"
        )
        answer_parts.append(src)

    elif task_type == "multilingual":
        answer_parts.append(
            f"**Traffic Regulation – {country} ({lang})**\n\n"
        )
        if title:
            answer_parts.append(f"**Section:** {title}\n\n")
        if text:
            answer_parts.append(f"**Original ({lang}):**\n> {text[:500].strip()}\n\n")
            answer_parts.append(f"{src}\n\n")
        answer_parts.append(
            f"**Explanation ({lang}):** This regulation from the {country} driving "
            f"handbook addresses requirements specific to {vtype} drivers. "
            f"Understanding the regulation in {lang} is critical for drivers operating "
            f"in {country}, where signage and enforcement use this language.\n\n"
        )
        answer_parts.append(
            f"**English Summary:** The {country} regulation under \"{title}\" establishes "
            f"rules for {vtype} operators. Key terms should be understood in the original "
            f"{lang} to avoid misinterpretation during driving or examinations.\n\n"
        )
        answer_parts.append(src)

    elif task_type == "edge_case":
        answer_parts.append(
            f"**Edge Case Analysis – {country} ({vtype})**\n\n"
        )
        if title:
            answer_parts.append(f"**Section:** {title}\n\n")
        if text:
            answer_parts.append(f"**Regulation:**\n> {text[:500].strip()}\n\n")
            answer_parts.append(f"{src}\n\n")
        answer_parts.append(
            f"**Identified Edge Cases:**\n\n"
            f"1. **Weather Conditions:** Adverse weather (heavy rain, snow, fog) may "
            f"require behaviour beyond the minimum stated in \"{title}\". For {vtype} "
            f"vehicles, stopping distances increase significantly.\n\n"
            f"2. **Vehicle-Specific Issues:** {vtype.capitalize()} vehicles have unique "
            f"handling characteristics that interact with this rule — wider turning radius, "
            f"longer braking distance, or visibility limitations.\n\n"
            f"3. **Conflicting Regulations:** This rule may intersect with temporary "
            f"traffic orders, construction zones, or police directions. In {country}, "
            f"police directions override posted rules.\n\n"
            f"4. **Emergency Situations:** Emergency vehicles (ambulance, fire, police) "
            f"may be exempt, but other drivers must still yield appropriately.\n\n"
            f"5. **Common Mistakes:** Drivers frequently misapply this rule when "
            f"encountering unfamiliar road layouts or faded signage.\n\n"
        )
        answer_parts.append(src)

    elif task_type == "test":
        answer_parts.append(
            f"**{country} Driving Examination – {vtype.capitalize()}**\n\n"
        )
        if title:
            answer_parts.append(f"**Topic:** {title}\n\n")
        answer_parts.append(f"**Question:**\n")
        answer_parts.append(
            f"According to the {country} driving handbook section \"{title}\", "
            f"which of the following is the correct procedure?\n\n"
            f"A) Ignore the regulation if road conditions seem safe\n"
            f"B) Follow the regulation exactly as specified in the handbook\n"
            f"C) Use personal judgement to determine when the rule applies\n"
            f"D) Only comply when other vehicles are present\n\n"
        )
        answer_parts.append(
            f"**Correct Answer: B)**\n\n"
            f"**Explanation:** The {country} highway code, under \"{title}\", "
            f"requires strict compliance regardless of perceived conditions. "
            f"The regulation states:\n> {text[:300].strip()}\n\n"
            f"{src}\n\n"
            f"Option A is dangerous — subjective assessment of safety does not override law. "
            f"Option C is incorrect because traffic rules are not discretionary. "
            f"Option D is wrong because regulations apply at all times, not only when "
            f"other traffic is present.\n\n"
        )
        answer_parts.append(
            f"**Exam Tip:** Questions on this topic test whether candidates understand "
            f"that traffic rules in {country} are absolute requirements, not suggestions. "
            f"Pay attention to specific distances, time limits, and exceptions.\n\n"
        )
        answer_parts.append(src)

    elif task_type == "vision":
        answer_parts.append(
            f"**Visual Traffic Analysis – {country}**\n\n"
        )
        if title:
            answer_parts.append(f"**Handbook Section:** {title}\n\n")
        answer_parts.append(
            f"**Image Analysis:**\n"
            f"The image from the {country} driving handbook ({vtype}, {lang}) "
            f"illustrates a key traffic concept covered under \"{title}\".\n\n"
            f"**Visual Elements Identified:**\n"
            f"- Traffic signs, road markings, or driving scenarios relevant to {vtype} operation\n"
            f"- Colour coding and symbol conventions used in {country}\n\n"
        )
        if text:
            answer_parts.append(
                f"**Accompanying Text:**\n> {text[:400].strip()}\n\n"
                f"{src}\n\n"
            )
        answer_parts.append(
            f"**Correct Driver Response:** Based on the visual cue and the "
            f"\"{title}\" section, the driver must identify the sign/marking and "
            f"respond according to {country} traffic law. Failure to recognise this "
            f"element is a common test failure point.\n\n"
        )
        answer_parts.append(
            f"**Follow-up Question:** Can you describe what would happen if a driver "
            f"failed to observe and act on this sign/marking in {country}?\n\n"
        )
        answer_parts.append(src)

    return think_block + "".join(answer_parts)


# ---------------------------------------------------------------------------
#  Image helpers (fast – no file open, just existence check)
# ---------------------------------------------------------------------------

def _fix_img_path(raw_path: str, data_root: str) -> str:
    """Normalise Windows backslashes and resolve to absolute path."""
    rel = raw_path.replace("\\", "/")
    return os.path.join(data_root, rel)


def _valid_image(path: str) -> bool:
    """Fast check: file exists and has an image extension."""
    if not os.path.isfile(path):
        return False
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))


# ---------------------------------------------------------------------------
#  Conversation builder
# ---------------------------------------------------------------------------

def build_conversation(
    entry: Dict,
    task_type: str,
    data_root: str,
    use_image: bool = False,
) -> Optional[Dict]:
    """Convert one handbook entry into a Qwen3-VL chat conversation dict.

    Returns None if the entry is unusable (empty text, missing image, etc.).
    """
    text = (entry.get("text") or "").strip()
    title = (entry.get("title") or "").strip()
    if not text and not title:
        return None

    fmt = {
        "country": entry.get("country", "Unknown"),
        "vehicle_type": entry.get("vehicle_type", "general"),
        "language": entry.get("language", "English"),
        "title": title,
        "text": text,
        "caption": "",
    }

    # Pick template
    templates = {
        "rule": RULE_TEMPLATES,
        "multilingual": MULTILINGUAL_TEMPLATES,
        "edge_case": EDGE_CASE_TEMPLATES,
        "test": TEST_TEMPLATES,
        "vision": VISION_TEMPLATES,
    }.get(task_type, RULE_TEMPLATES)

    prompt_text = random.choice(templates).format(**fmt)

    # Build user content
    user_content = []

    if use_image and entry.get("imgs"):
        img_info = entry["imgs"][0]
        img_path = _fix_img_path(img_info.get("img_path", ""), data_root)
        if not _valid_image(img_path):
            use_image = False
        else:
            user_content.append({"type": "image", "image": f"file://{img_path}"})
            cap = img_info.get("figure_caption", [])
            if cap:
                fmt["caption"] = cap[0] if isinstance(cap, list) else str(cap)
                prompt_text = random.choice(VISION_TEMPLATES).format(**fmt)

    user_content.append({"type": "text", "text": prompt_text})

    # Build CoT + grounded assistant answer
    answer = _build_cot_answer(entry, task_type)

    conversation = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "_meta": {
            "country": entry.get("country", ""),
            "vehicle_type": entry.get("vehicle_type", ""),
            "language": entry.get("language", ""),
            "task_type": task_type,
            "has_image": use_image,
        },
    }

    return conversation


# ---------------------------------------------------------------------------
#  Dataset builder
# ---------------------------------------------------------------------------

def load_handbook(data_json: str) -> List[Dict]:
    """Load the driving handbook JSON."""
    with open(data_json, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_task_type(entry: Dict) -> str:
    """Heuristically assign a task type to an entry."""
    lang = entry.get("language", "English")
    text = (entry.get("text") or "").lower()
    vehicle = entry.get("vehicle_type", "")

    # Vision entries (have images)
    if entry.get("imgs"):
        return "vision"

    # Multilingual (non-English)
    if lang != "English":
        return "multilingual"

    # Edge cases: truck/moto or mixed vehicle types, or text mentioning exceptions
    edge_keywords = ["exception", "emergency", "special", "prohibited", "unusual",
                     "caution", "warning", "penalty", "fine", "demerit"]
    if vehicle in ("truck", "moto") or any(kw in text for kw in edge_keywords):
        if random.random() < 0.4:
            return "edge_case"

    # Test-style: entries with "question", "test", "exam" in title/text
    test_keywords = ["question", "test", "exam", "quiz", "answer", "correct"]
    title_lower = (entry.get("title") or "").lower()
    if any(kw in title_lower or kw in text for kw in test_keywords):
        if random.random() < 0.6:
            return "test"

    # Default: rule reasoning
    return "rule"


def build_dataset(
    data_json: str,
    data_root: str,
    max_samples: int = 0,
    seed: int = 42,
    val_ratio: float = 0.05,
    include_images: bool = True,
) -> dict:
    """Build train/val splits from driving handbook data.

    Returns dict with 'train' and 'val' keys, each a list of conversation dicts.
    """
    random.seed(seed)
    entries = load_handbook(data_json)
    print(f"  Loaded {len(entries)} handbook entries from {data_json}")

    conversations = []
    skipped = 0
    task_counts = {}

    for i, entry in enumerate(entries):
        task_type = assign_task_type(entry)
        use_img = include_images and task_type == "vision" and bool(entry.get("imgs"))

        conv = build_conversation(entry, task_type, data_root, use_image=use_img)
        if conv is None:
            skipped += 1
            continue

        conversations.append(conv)
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

        if (i + 1) % 10000 == 0:
            print(f"    processed {i+1}/{len(entries)}...")

    print(f"  Built {len(conversations)} conversations ({skipped} skipped)")
    print(f"  Task distribution: {json.dumps(task_counts, indent=2)}")

    # Shuffle and split
    random.shuffle(conversations)
    if max_samples > 0:
        conversations = conversations[:max_samples]

    n_val = max(1, int(len(conversations) * val_ratio))
    val_data = conversations[:n_val]
    train_data = conversations[n_val:]

    print(f"  Train: {len(train_data)}  Val: {len(val_data)}")
    return {"train": train_data, "val": val_data}


def save_dataset(dataset: dict, output_dir: str):
    """Save as JSONL files for easy loading."""
    os.makedirs(output_dir, exist_ok=True)
    for split, convs in dataset.items():
        path = os.path.join(output_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for c in convs:
                out = {"messages": c["messages"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  Saved {len(convs)} samples to {path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SafetyVLM teacher dataset (CoT + Grounding)")
    parser.add_argument("--data_json",  default="/scratch/jnolas77/SafetyVLM/driving_handbook_data/data.json")
    parser.add_argument("--data_root",  default="/scratch/jnolas77/SafetyVLM/driving_handbook_data")
    parser.add_argument("--output_dir", default="/scratch/jnolas77/SafetyVLM/Qwen-3-VL/data")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_images",  action="store_true")
    args = parser.parse_args()

    ds = build_dataset(
        args.data_json, args.data_root,
        max_samples=args.max_samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
        include_images=not args.no_images,
    )
    save_dataset(ds, args.output_dir)
    print("Done.")
