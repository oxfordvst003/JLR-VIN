"""
Resplit Data into 60:20:20 (Train:Val:Test)
============================================
Consolidates all images into one pool, then redistributes with proper ratios.
Uses random seed for reproducibility.
"""

import os
import shutil
import random
from pathlib import Path

# Import centralized config
try:
    from config import DATA_ROOT, SEED
    DATA_ROOT = str(DATA_ROOT)
except ImportError:
    DATA_ROOT = r"C:\JLR_VIN_2549\Data"  # fallback
    SEED = 42

SPLITS = ["train", "val", "test"]

# Target ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20


def collect_all_files():
    """Collect all image-label pairs from all splits."""
    all_pairs = []
    
    for split in SPLITS:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        label_dir = os.path.join(DATA_ROOT, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            
            img_path = os.path.join(img_dir, fname)
            label_name = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
            
            if os.path.exists(label_path):
                all_pairs.append({
                    "img_name": fname,
                    "label_name": label_name,
                    "img_path": img_path,
                    "label_path": label_path,
                    "original_split": split
                })
    
    return all_pairs


def create_backup():
    """Create backup of current data structure."""
    backup_dir = os.path.join(DATA_ROOT, "_backup")
    if os.path.exists(backup_dir):
        print(f"Backup already exists at {backup_dir}")
        return
    
    print("Creating backup...")
    for split in SPLITS:
        src_img = os.path.join(DATA_ROOT, split, "images")
        src_lbl = os.path.join(DATA_ROOT, split, "labels")
        dst_img = os.path.join(backup_dir, split, "images")
        dst_lbl = os.path.join(backup_dir, split, "labels")
        
        if os.path.exists(src_img):
            shutil.copytree(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copytree(src_lbl, dst_lbl)
    
    print(f"Backup created at: {backup_dir}")


def clear_splits():
    """Clear all split directories."""
    for split in SPLITS:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        label_dir = os.path.join(DATA_ROOT, split, "labels")
        
        for d in [img_dir, label_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))


def distribute_files(all_pairs):
    """Shuffle and distribute files into splits."""
    random.seed(SEED)
    random.shuffle(all_pairs)
    
    total = len(all_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits_assignment = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:]
    }
    
    return splits_assignment


def copy_files_to_splits(splits_assignment):
    """Copy files to their new split directories."""
    for split, pairs in splits_assignment.items():
        img_dir = os.path.join(DATA_ROOT, split, "images")
        label_dir = os.path.join(DATA_ROOT, split, "labels")
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        for pair in pairs:
            # Copy image
            dst_img = os.path.join(img_dir, pair["img_name"])
            shutil.copy2(pair["img_path"], dst_img)
            
            # Copy label
            dst_label = os.path.join(label_dir, pair["label_name"])
            shutil.copy2(pair["label_path"], dst_label)
        
        print(f"  {split}: {len(pairs)} files")


def main():
    print("=" * 60)
    print("RESPLIT DATA INTO 60:20:20")
    print("=" * 60)
    
    # 1. Collect all files
    print("\n1. Collecting all image-label pairs...")
    all_pairs = collect_all_files()
    print(f"   Found {len(all_pairs)} unique image-label pairs")
    
    # 2. Create backup
    print("\n2. Creating backup...")
    create_backup()
    
    # 3. Calculate new splits
    print("\n3. Calculating new split sizes...")
    splits_assignment = distribute_files(all_pairs)
    
    total = len(all_pairs)
    print(f"   Target: Train={int(total*0.6)}, Val={int(total*0.2)}, Test={int(total*0.2)}")
    print(f"   Actual: Train={len(splits_assignment['train'])}, Val={len(splits_assignment['val'])}, Test={len(splits_assignment['test'])}")
    
    # 4. Clear existing splits
    print("\n4. Clearing existing split directories...")
    clear_splits()
    
    # 5. Copy files to new locations
    print("\n5. Copying files to new splits...")
    copy_files_to_splits(splits_assignment)
    
    # 6. Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    for split in SPLITS:
        img_count = len(os.listdir(os.path.join(DATA_ROOT, split, "images")))
        label_count = len(os.listdir(os.path.join(DATA_ROOT, split, "labels")))
        ratio = img_count / total * 100
        print(f"  {split:6}: {img_count:4} images, {label_count:4} labels ({ratio:.1f}%)")
    
    print("\n✅ Resplit complete!")
    print(f"   Seed used: {SEED} (for reproducibility)")
    print(f"   Backup at: {os.path.join(DATA_ROOT, '_backup')}")


if __name__ == "__main__":
    main()
