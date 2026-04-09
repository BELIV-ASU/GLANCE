#!/usr/bin/env python3
"""
Extract Waymo Front Camera Images and Prepare for GLANCE Training

This script:
1. Reads the downloaded Waymo TFRecords from HuggingFace cache
2. Extracts ONLY FRONT camera images
3. Saves them as JPEGs organized by split
4. Generates annotations JSON for GLANCE training
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup all necessary paths."""
    waymo_cache = Path("/scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3")
    waymo_blobs = waymo_cache / "blobs"
    
    # Output directories
    waymo_output = Path("/scratch/rbaskar5/Dataset/waymo_front")
    train_dir = waymo_output / "training" / "camera_FRONT"
    val_dir = waymo_output / "validation" / "camera_FRONT"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_dir = waymo_output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'cache': waymo_cache,
        'blobs': waymo_blobs,
        'output': waymo_output,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'annotations_dir': annotations_dir,
    }

def extract_front_camera_images(paths: Dict[str, Path], max_files: int = 0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract FRONT camera images from TFRecords.
    
    Returns:
        Dict with 'train' and 'val' keys containing annotation lists
    """
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
    except ImportError:
        logger.error("⚠ TensorFlow and waymo-open-dataset-tf-2-12-0 not installed!")
        logger.error("Run: pip install tensorflow waymo-open-dataset-tf-2-12-0")
        sys.exit(1)
    
    import io
    from PIL import Image
    
    logger.info("Starting extraction of FRONT camera images...")
    
    # Camera ID mapping
    FRONT_CAM_ID = 1
    
    annotations = {'train': [], 'val': []}
    frame_count = {'train': 0, 'val': 0}
    
    # Find all tfrecord files
    tfrecord_files = sorted(glob.glob(str(paths['blobs'] / "*")))
    if max_files is not None and max_files > 0:
        tfrecord_files = tfrecord_files[:max_files]
    
    logger.info(f"Found {len(tfrecord_files)} files to process")
    
    for file_idx, tf_file in enumerate(tfrecord_files):
        logger.info(f"[{file_idx+1}/{len(tfrecord_files)}] Processing {Path(tf_file).name}...")
        
        try:
            dataset = tf.data.TFRecordDataset(tf_file, compression_type="")
            
            for frame_idx, raw_record in enumerate(dataset):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(raw_record.numpy())
                
                # Extract FRONT camera image
                for cam_image in frame.images:
                    if cam_image.name != FRONT_CAM_ID:
                        continue
                    
                    try:
                        # Decode JPEG
                        pil_img = Image.open(io.BytesIO(cam_image.image)).convert("RGB")
                        
                        # Determine split (rough: first 80% train, last 20% val)
                        split_phase = file_idx / len(tfrecord_files)
                        split = 'train' if split_phase < 0.8 else 'val'
                        
                        # Save image
                        img_filename = f"frame_{frame_idx:06d}_file_{file_idx:04d}.jpg"
                        img_path = paths[f'{split}_dir'] / img_filename
                        pil_img.save(img_path, format='JPEG', quality=95)
                        
                        # Create annotation
                        annotation = {
                            "image_path": f"{'training' if split == 'train' else 'validation'}/camera_FRONT/{img_filename}",
                            "camera": "FRONT",
                            "prompt": (
                                "Analyze this street scene from an autonomous vehicle's perspective. "
                                "Describe all visible objects including: vehicles (cars, trucks, buses), "
                                "pedestrians, cyclists, traffic signs, signals, and road conditions. "
                                "Identify any traffic violations or unusual events."
                            ),
                            "answer": (
                                "This is a front-camera view from the Waymo dataset. "
                                "[Replace with detailed scene understanding labels during annotation phase]"
                            ),
                            "metadata": {
                                "dataset": "waymo",
                                "camera": "FRONT",
                                "timestamp": frame.timestamp_micros,
                                "file_idx": file_idx,
                                "frame_idx": frame_idx,
                            }
                        }
                        
                        annotations[split].append(annotation)
                        frame_count[split] += 1
                        
                        if (frame_count['train'] + frame_count['val']) % 100 == 0:
                            logger.info(f"  Extracted {frame_count['train'] + frame_count['val']} frames total")
                    
                    except Exception as e:
                        logger.warning(f"Failed to process frame: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error processing file {tf_file}: {e}")
            continue
    
    logger.info(f"\n✓ Extraction complete!")
    logger.info(f"  Train frames: {frame_count['train']}")
    logger.info(f"  Val frames:   {frame_count['val']}")
    
    return annotations

def save_annotations(annotations: Dict[str, List[Dict]], paths: Dict[str, Path]):
    """Save annotations to JSON files."""
    for split in ['train', 'val']:
        json_path = paths['annotations_dir'] / f"{split}.json"
        with open(json_path, 'w') as f:
            json.dump(annotations[split], f, indent=2)
        logger.info(f"✓ Saved {len(annotations[split])} {split} annotations to {json_path}")

def update_training_config(paths: Dict[str, Path]):
    """Update waymo_finetune_config.yaml to use extracted data."""
    config_path = Path("/scratch/rbaskar5/GLANCE/configs/waymo_finetune_config.yaml")
    
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}")
        return
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update paths to use front camera extracted data
    content_updated = content.replace(
        'waymo_root: "/scratch/rbaskar5/Dataset/waymo"',
        f'waymo_root: "{paths["output"]}"'
    )
    
    content_updated = content_updated.replace(
        'annotations_json: "/scratch/rbaskar5/Dataset/waymo/annotations/train.json"',
        f'annotations_json: "{paths["annotations_dir"]}/train.json"'
    )
    
    content_updated = content_updated.replace(
        'val_annotations_json: "/scratch/rbaskar5/Dataset/waymo/annotations/val.json"',
        f'val_annotations_json: "{paths["annotations_dir"]}/val.json"'
    )
    
    with open(config_path, 'w') as f:
        f.write(content_updated)
    
    logger.info(f"✓ Updated config file: {config_path}")
    logger.info(f"  Waymo root: {paths['output']}")
    logger.info(f"  Train annotations: {paths['annotations_dir']}/train.json")
    logger.info(f"  Val annotations: {paths['annotations_dir']}/val.json")

def main():
    logger.info("="*70)
    logger.info("WAYMO FRONT CAMERA EXTRACTION FOR GLANCE")
    logger.info("="*70)
    
    # Setup paths
    paths = setup_paths()
    logger.info(f"Output directory: {paths['output']}")
    
    # Check if source data exists
    if not paths['blobs'].exists():
        logger.error(f"Waymo TFRecords not found at {paths['blobs']}")
        logger.error("Make sure you've downloaded the Waymo dataset to HuggingFace cache")
        sys.exit(1)
    
    # Extract images from all available files.
    max_files_limit = 0
    annotations = extract_front_camera_images(paths, max_files=max_files_limit)
    
    # Save annotations
    save_annotations(annotations, paths)
    
    # Update config
    update_training_config(paths)
    
    logger.info("\n" + "="*70)
    logger.info("✓ SETUP COMPLETE")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("1. Review the extracted images and annotations:")
    logger.info(f"   ls -lh {paths['train_dir']}")
    logger.info(f"   cat {paths['annotations_dir']}/train.json | head")
    logger.info("\n2. Start training with Qwen2.5-VL-7B:")
    logger.info("   cd /scratch/rbaskar5/GLANCE")
    logger.info("   source /scratch/rbaskar5/set.bash")
    logger.info("   ./scripts/train_waymo.sh qwen2.5-vl-7b")
    logger.info("\n3. Monitor training:")
    logger.info(f"   tail -f /scratch/rbaskar5/GLANCE/checkpoints/waymo_finetune/training_log.txt")

if __name__ == "__main__":
    main()
