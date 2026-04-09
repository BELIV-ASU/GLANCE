#!/usr/bin/env python3
"""
Verify GLANCE Waymo + Qwen setup is complete and ready to train
"""

import os
import sys
from pathlib import Path

def check_item(name, condition, details=""):
    status = "✓" if condition else "✗"
    print(f"  {status} {name:50s} {details}")
    return condition

def main():
    print("\n" + "="*80)
    print("GLANCE WAYMO + QWEN SETUP VERIFICATION")
    print("="*80 + "\n")
    
    all_ok = True
    
    # 1. Check Waymo dataset
    print("1. Waymo Dataset")
    waymo_cache = Path("/scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3")
    waymo_blobs = waymo_cache / "blobs"
    
    if waymo_cache.exists():
        all_ok &= check_item("Dataset cache dir exists", True, str(waymo_cache))
    else:
        all_ok &= check_item("Dataset cache dir exists", False, f"NOT FOUND: {waymo_cache}")
    
    if waymo_blobs.exists():
        num_files = len(list(waymo_blobs.iterdir()))
        all_ok &= check_item("Waymo blobs", True, f"{num_files} files")
    else:
        all_ok &= check_item("Waymo blobs", False, "NOT FOUND")
    
    # 2. Check GLANCE scripts
    print("\n2. GLANCE Scripts")
    glance_root = Path("/scratch/rbaskar5/GLANCE")
    
    scripts_to_check = [
        ("Preparation script", glance_root / "scripts" / "prepare_waymo_front_camera.py"),
        ("Training script", glance_root / "scripts" / "train_waymo.sh"),
        ("Setup orchestrator", glance_root / "setup_glance_waymo_qwen.sh"),
        ("Train Waymo module", glance_root / "src" / "train_waymo.py"),
        ("Dataset loader", glance_root / "src" / "waymo_dataset.py"),
        ("VLM factory", glance_root / "src" / "vlm_factory.py"),
    ]
    
    for name, path in scripts_to_check:
        exists = path.exists()
        all_ok &= check_item(name, exists, str(path) if exists else "NOT FOUND")
    
    # 3. Check Configuration
    print("\n3. Configuration")
    config_path = glance_root / "configs" / "waymo_finetune_config.yaml"
    
    if config_path.exists():
        check_item("Config file", True, str(config_path))
        with open(config_path) as f:
            content = f.read()
            has_qwen = "qwen2.5-vl-7b" in content
            all_ok &= check_item("Uses Qwen2.5-VL-7B", has_qwen)
            
            has_lora =  "lora:" in content
            all_ok &= check_item("LoRA config present", has_lora)
    else:
        all_ok &= check_item("Config file", False, "NOT FOUND")
    
    # 4. Check Documentation
    print("\n4. Documentation")
    docs = [
        ("Setup guide", glance_root / "GLANCE_WAYMO_QWEN_SETUP.md"),
        ("Quick start", glance_root / "QUICK_START.md"),
    ]
    
    for name, path in docs:
        exists = path.exists()
        all_ok &= check_item(name, exists, str(path) if exists else "NOT FOUND")
    
    # 5. Check HF environment
    print("\n5. Python Environment")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_ok &= check_item("PyTorch installed", True, f"v{torch.__version__}")
        all_ok &= check_item("CUDA available", cuda_available)
    except ImportError:
        all_ok &= check_item("PyTorch installed", False, "Run: pip install torch")
    
    try:
        import transformers
        all_ok &= check_item("Transformers installed", True, f"v{transformers.__version__}")
    except ImportError:
        all_ok &= check_item("Transformers installed", False, "Run: pip install transformers")
    
    try:
        from peft import PeftModel
        all_ok &= check_item("PEFT (LoRA) installed", True, "")
    except ImportError:
        all_ok &= check_item("PEFT (LoRA) installed", False, "Run: pip install peft")
    
    try:
        from waymo_open_dataset import dataset_pb2
        all_ok &= check_item("Waymo SDK installed", True, "waymo-open-dataset-tf-2-12-0")
    except ImportError:
        all_ok &= check_item("Waymo SDK installed", False, "Run: pip install waymo-open-dataset-tf-2-12-0")
    
    # 6. Check file structure
    print("\n6. Project Structure")
    dataset_dir = Path("/scratch/rbaskar5/Dataset")
    all_ok &= check_item("Dataset directory", dataset_dir.exists(), str(dataset_dir))
    
    # 7. Summary
    print("\n" + "="*80)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("  1. cd /scratch/rbaskar5/GLANCE")
        print("  2. source /scratch/rbaskar5/set.bash")
        print("  3. python scripts/prepare_waymo_front_camera.py  # Extract data (15-30 min)")
        print("  4. bash scripts/train_waymo.sh qwen2.5-vl-7b    # Train (3-5 hours)")
        print("\nOr run everything at once:")
        print("  bash setup_glance_waymo_qwen.sh both")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before training")
        print("\nFor details, see:")
        print("  - /scratch/rbaskar5/GLANCE/QUICK_START.md")
        print("  - /scratch/rbaskar5/GLANCE/GLANCE_WAYMO_QWEN_SETUP.md")
        return 1
    
    print("="*80 + "\n")

if __name__ == "__main__":
    sys.exit(main())
