"""
Data Split Validation & Deduplication Script
=============================================
Checks for idempotency across train/val/test splits and optionally fixes duplicates.

Usage:
    python validate_data_splits.py          # Check only (no changes)
    python validate_data_splits.py --fix    # Remove duplicates (keep in train, remove from val/test)
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

# Import centralized config
try:
    from config import DATA_ROOT
    DATA_ROOT = str(DATA_ROOT)
except ImportError:
    DATA_ROOT = r"C:\JLR_VIN_2549\Data"  # fallback

SPLITS = ["train", "val", "test"]


def get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file content."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_all_images():
    """Collect all image files with their metadata."""
    all_files = []
    for split in SPLITS:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        if not os.path.exists(img_dir):
            print(f"[WARN] Directory not found: {img_dir}")
            continue
        for fname in os.listdir(img_dir):
            fpath = os.path.join(img_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                h = get_file_hash(fpath)
                all_files.append({
                    "split": split,
                    "name": fname,
                    "path": fpath,
                    "hash": h,
                })
    return all_files


def find_duplicates(all_files):
    """Find duplicate images across splits."""
    hash_to_files = defaultdict(list)
    for f in all_files:
        hash_to_files[f["hash"]].append(f)
    
    # Only keep hashes that appear in multiple splits
    duplicates = {}
    for h, files in hash_to_files.items():
        splits_present = set(f["split"] for f in files)
        if len(splits_present) > 1:
            duplicates[h] = files
    
    return duplicates


def check_label_consistency(all_files):
    """Check if all images have corresponding label files."""
    missing_labels = []
    for f in all_files:
        label_path = f["path"].replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        if not os.path.exists(label_path):
            missing_labels.append(f)
    return missing_labels


def print_report(all_files, duplicates, missing_labels):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("DATA SPLIT VALIDATION REPORT")
    print("=" * 60)
    
    # Per-split counts
    split_counts = defaultdict(int)
    for f in all_files:
        split_counts[f["split"]] += 1
    
    print("\n📊 SPLIT SIZES:")
    for split in SPLITS:
        print(f"   {split:6}: {split_counts[split]:4} images")
    print(f"   {'TOTAL':6}: {len(all_files):4} images")
    
    # Unique count
    unique_hashes = set(f["hash"] for f in all_files)
    print(f"\n   Unique images (by content): {len(unique_hashes)}")
    
    # Overlap analysis
    print("\n🔍 OVERLAP ANALYSIS:")
    
    split_hashes = {split: set() for split in SPLITS}
    for f in all_files:
        split_hashes[f["split"]].add(f["hash"])
    
    train_val = split_hashes["train"] & split_hashes["val"]
    train_test = split_hashes["train"] & split_hashes["test"]
    val_test = split_hashes["val"] & split_hashes["test"]
    
    print(f"   Train ∩ Val:  {len(train_val):3} duplicates")
    print(f"   Train ∩ Test: {len(train_test):3} duplicates")
    print(f"   Val ∩ Test:   {len(val_test):3} duplicates")
    
    total_overlap = len(train_val) + len(train_test) + len(val_test)
    
    # Label consistency
    print("\n📝 LABEL CONSISTENCY:")
    if missing_labels:
        print(f"   ⚠️  Missing label files: {len(missing_labels)}")
        for f in missing_labels[:5]:
            print(f"      [{f['split']}] {f['name']}")
        if len(missing_labels) > 5:
            print(f"      ... and {len(missing_labels) - 5} more")
    else:
        print("   ✅ All images have corresponding label files")
    
    # Final verdict
    print("\n" + "=" * 60)
    if total_overlap == 0 and not missing_labels:
        print("✅ DATA IS IDEMPOTENT: No issues found!")
    else:
        print("⚠️  ISSUES FOUND:")
        if total_overlap > 0:
            print(f"   - {total_overlap} duplicate images across splits")
        if missing_labels:
            print(f"   - {len(missing_labels)} missing label files")
    print("=" * 60)
    
    return total_overlap, len(missing_labels)


def fix_duplicates(duplicates, dry_run=False):
    """
    Remove duplicates from val/test, keeping them in train.
    Priority: train > val > test
    """
    print("\n🔧 FIXING DUPLICATES...")
    
    removed = {"val": 0, "test": 0}
    priority = {"train": 0, "val": 1, "test": 2}
    
    for h, files in duplicates.items():
        # Sort by priority (keep the one with lowest priority number)
        files_sorted = sorted(files, key=lambda x: priority[x["split"]])
        keep = files_sorted[0]
        remove = files_sorted[1:]
        
        for f in remove:
            img_path = f["path"]
            label_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            label_path = os.path.splitext(label_path)[0] + ".txt"
            
            if dry_run:
                print(f"   [DRY-RUN] Would remove: {f['split']}/{f['name']}")
            else:
                # Remove image
                if os.path.exists(img_path):
                    os.remove(img_path)
                # Remove label
                if os.path.exists(label_path):
                    os.remove(label_path)
                print(f"   Removed: {f['split']}/{f['name']}")
            
            removed[f["split"]] += 1
    
    print(f"\n   Removed from val:  {removed['val']}")
    print(f"   Removed from test: {removed['test']}")
    return removed


def main():
    fix_mode = "--fix" in sys.argv
    dry_run = "--dry-run" in sys.argv
    
    if fix_mode:
        print("🔧 FIX MODE: Will remove duplicates from val/test (keeping in train)")
        if dry_run:
            print("   (DRY-RUN: No actual changes will be made)")
    else:
        print("🔍 CHECK MODE: Validating data splits (use --fix to remove duplicates)")
    
    # Collect data
    print("\nScanning image files...")
    all_files = collect_all_images()
    
    print("Computing file hashes...")
    duplicates = find_duplicates(all_files)
    
    print("Checking label files...")
    missing_labels = check_label_consistency(all_files)
    
    # Report
    overlap_count, missing_count = print_report(all_files, duplicates, missing_labels)
    
    # Fix if requested
    if fix_mode and duplicates:
        fix_duplicates(duplicates, dry_run=dry_run)
        
        # Re-validate
        print("\n" + "=" * 60)
        print("POST-FIX VALIDATION")
        print("=" * 60)
        all_files = collect_all_images()
        duplicates = find_duplicates(all_files)
        missing_labels = check_label_consistency(all_files)
        print_report(all_files, duplicates, missing_labels)


if __name__ == "__main__":
    main()
