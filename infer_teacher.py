"""
infer_teacher.py  –  Run inference with the fine-tuned SafetyVLM Teacher (32B)

Usage:
  python infer_teacher.py --prompt "Explain the right-of-way rule at a roundabout in the UK"
  python infer_teacher.py --prompt "..." --image /path/to/traffic_sign.png
  python infer_teacher.py --interactive
"""

import os, sys, json, argparse, torch
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as ModelClass


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


def load_model(args):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {args.base_model}")

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
    )

    if args.adapter_path and os.path.isdir(args.adapter_path):
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )

    return model, processor


def generate(model, processor, prompt: str, image_path: str = None,
             max_new_tokens: int = 512, temperature: float = 0.7):
    """Generate a response."""
    user_content = []

    if image_path and os.path.isfile(image_path):
        from PIL import Image
        user_content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})

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

    # Decode only new tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def interactive_mode(model, processor, args):
    """Interactive chatbot loop."""
    print("\n  SafetyVLM Teacher – Interactive Mode")
    print("  Type 'quit' to exit, 'image:<path>' to attach an image\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        image_path = None
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0].replace("image:", "")
            user_input = parts[1] if len(parts) > 1 else "Describe this traffic sign or scenario."

        response = generate(model, processor, user_input, image_path,
                          max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print(f"\nTeacher: {response}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-32B")
    parser.add_argument("--adapter_path", default="/scratch/jnolas77/SafetyVLM/Qwen-3-VL/checkpoints/final_lora")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, processor = load_model(args)

    if args.interactive:
        interactive_mode(model, processor, args)
    elif args.prompt:
        response = generate(model, processor, args.prompt, args.image,
                          max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print(f"\n{response}")
    else:
        # Default demo prompts
        demos = [
            "Explain the right-of-way rules at a 4-way stop intersection in the USA.",
            "What are the key differences between UK and German motorway speed regulations?",
            "A truck driver in Canada encounters a school bus with flashing red lights. What must they do?",
        ]
        for p in demos:
            print(f"\n{'='*60}")
            print(f"Q: {p}")
            print(f"{'='*60}")
            response = generate(model, processor, p, max_new_tokens=args.max_new_tokens)
            print(f"\n{response}\n")


if __name__ == "__main__":
    main()
