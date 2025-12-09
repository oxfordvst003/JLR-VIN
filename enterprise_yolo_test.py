"""
================================================================================
JLR VIN YOLO Model - Enterprise Testing & Validation Plan
================================================================================
Purpose: Test YOLO model performance after data cleaning and 60:20:20 resplit
Author: OVST Data Science Team
Date: December 2024

This script provides a comprehensive testing pipeline:
1. Data validation & integrity checks
2. Model inference on all splits
3. Performance metrics calculation
4. Detailed reporting & visualization
5. Export results for stakeholder review
================================================================================
"""

import os
import sys
import json
import csv
import time
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Ensure we can import from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_ROOT, DATA_YAML,
    TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES,
    TRAIN_LABELS, VAL_LABELS, TEST_LABELS,
    YOLO_BEST_FIXED_SEED, YOLO_BEST_OPTUNA,
    CLASS_NAMES, NUM_CLASSES, IMGSZ
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TestConfig:
    """Central configuration for the testing pipeline."""
    
    # Output directory for results
    RESULTS_DIR = PROJECT_ROOT / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Models to test
    MODELS = {
        "fixed_seed": YOLO_BEST_FIXED_SEED,
        "optuna": YOLO_BEST_OPTUNA,
    }
    
    # Inference settings
    CONFIDENCE_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.5
    IMAGE_SIZE = IMGSZ
    
    # Test modes
    RUN_DATA_VALIDATION = True
    RUN_INFERENCE_TEST = True
    RUN_METRICS_ANALYSIS = True
    RUN_VISUALIZATION = True
    EXPORT_DETAILED_RESULTS = True
    
    # Batch settings
    BATCH_SIZE = 16
    
    @classmethod
    def setup(cls):
        """Create output directories."""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.RESULTS_DIR / "plots").mkdir(exist_ok=True)
        (cls.RESULTS_DIR / "reports").mkdir(exist_ok=True)
        (cls.RESULTS_DIR / "predictions").mkdir(exist_ok=True)
        print(f"📁 Results will be saved to: {cls.RESULTS_DIR}")


# ============================================================================
# PHASE 1: DATA VALIDATION
# ============================================================================

class DataValidator:
    """Validates data integrity before model testing."""
    
    def __init__(self):
        self.splits = {
            "train": (TRAIN_IMAGES, TRAIN_LABELS),
            "val": (VAL_IMAGES, VAL_LABELS),
            "test": (TEST_IMAGES, TEST_LABELS),
        }
        self.validation_results = {}
    
    def run_all_checks(self) -> Dict:
        """Run all validation checks."""
        print("\n" + "="*60)
        print("PHASE 1: DATA VALIDATION")
        print("="*60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check 1: Split sizes
        results["checks"]["split_sizes"] = self._check_split_sizes()
        
        # Check 2: Label consistency
        results["checks"]["label_consistency"] = self._check_label_consistency()
        
        # Check 3: No duplicates across splits
        results["checks"]["no_duplicates"] = self._check_no_duplicates()
        
        # Check 4: Label format validation
        results["checks"]["label_format"] = self._check_label_format()
        
        # Check 5: Class distribution
        results["checks"]["class_distribution"] = self._check_class_distribution()
        
        # Summary
        all_passed = all(
            check.get("passed", False) 
            for check in results["checks"].values()
        )
        results["all_passed"] = all_passed
        
        self.validation_results = results
        return results
    
    def _check_split_sizes(self) -> Dict:
        """Verify split sizes match expected 60:20:20 ratio."""
        print("\n📊 Checking split sizes...")
        
        sizes = {}
        for split, (img_dir, _) in self.splits.items():
            count = len(list(Path(img_dir).glob("*.[jJ][pP][gG]")) + 
                       list(Path(img_dir).glob("*.[pP][nN][gG]")))
            sizes[split] = count
        
        total = sum(sizes.values())
        ratios = {k: v/total*100 for k, v in sizes.items()}
        
        # Check if ratios are approximately 60:20:20 (allow 5% tolerance)
        expected = {"train": 60, "val": 20, "test": 20}
        passed = all(abs(ratios[k] - expected[k]) < 5 for k in expected)
        
        result = {
            "passed": passed,
            "sizes": sizes,
            "ratios": {k: f"{v:.1f}%" for k, v in ratios.items()},
            "total": total
        }
        
        status = "✅" if passed else "⚠️"
        print(f"   {status} Train: {sizes['train']} ({ratios['train']:.1f}%)")
        print(f"   {status} Val: {sizes['val']} ({ratios['val']:.1f}%)")
        print(f"   {status} Test: {sizes['test']} ({ratios['test']:.1f}%)")
        print(f"   Total: {total} images")
        
        return result
    
    def _check_label_consistency(self) -> Dict:
        """Verify all images have corresponding label files."""
        print("\n📝 Checking label consistency...")
        
        missing = []
        for split, (img_dir, label_dir) in self.splits.items():
            for img_path in Path(img_dir).iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_path = Path(label_dir) / (img_path.stem + ".txt")
                    if not label_path.exists():
                        missing.append((split, img_path.name))
        
        passed = len(missing) == 0
        result = {
            "passed": passed,
            "missing_count": len(missing),
            "missing_files": missing[:10]  # First 10 only
        }
        
        if passed:
            print("   ✅ All images have corresponding label files")
        else:
            print(f"   ❌ Missing {len(missing)} label files")
            for split, fname in missing[:5]:
                print(f"      [{split}] {fname}")
        
        return result
    
    def _check_no_duplicates(self) -> Dict:
        """Verify no duplicate images across splits (by content hash)."""
        print("\n🔍 Checking for duplicates across splits...")
        
        hashes = defaultdict(list)
        
        for split, (img_dir, _) in self.splits.items():
            for img_path in Path(img_dir).iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    with open(img_path, 'rb') as f:
                        h = hashlib.md5(f.read()).hexdigest()
                    hashes[h].append((split, img_path.name))
        
        duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
        duplicate_count = sum(len(f) - 1 for f in duplicates.values())
        
        passed = len(duplicates) == 0
        result = {
            "passed": passed,
            "duplicate_groups": len(duplicates),
            "total_duplicates": duplicate_count,
        }
        
        if passed:
            print("   ✅ No duplicates found across splits")
        else:
            print(f"   ❌ Found {duplicate_count} duplicates in {len(duplicates)} groups")
        
        return result
    
    def _check_label_format(self) -> Dict:
        """Validate YOLO label format."""
        print("\n📋 Checking label format...")
        
        errors = []
        total_labels = 0
        total_boxes = 0
        
        for split, (_, label_dir) in self.splits.items():
            for label_path in Path(label_dir).glob("*.txt"):
                total_labels += 1
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        total_boxes += 1
                        parts = line.split()
                        
                        # Check format: class_id x_center y_center width height
                        if len(parts) != 5:
                            errors.append((split, label_path.name, line_num, "Invalid column count"))
                            continue
                        
                        try:
                            cls_id = int(parts[0])
                            coords = [float(p) for p in parts[1:]]
                            
                            # Validate class ID
                            if cls_id < 0 or cls_id >= NUM_CLASSES:
                                errors.append((split, label_path.name, line_num, f"Invalid class ID: {cls_id}"))
                            
                            # Validate coordinates (should be 0-1)
                            if not all(0 <= c <= 1 for c in coords):
                                errors.append((split, label_path.name, line_num, "Coordinates out of range"))
                                
                        except ValueError as e:
                            errors.append((split, label_path.name, line_num, str(e)))
        
        passed = len(errors) == 0
        result = {
            "passed": passed,
            "total_labels": total_labels,
            "total_boxes": total_boxes,
            "error_count": len(errors),
            "errors": errors[:10]
        }
        
        if passed:
            print(f"   ✅ All {total_labels} label files valid ({total_boxes} bounding boxes)")
        else:
            print(f"   ❌ Found {len(errors)} format errors")
        
        return result
    
    def _check_class_distribution(self) -> Dict:
        """Analyze class distribution across splits."""
        print("\n📈 Analyzing class distribution...")
        
        distributions = {}
        
        for split, (_, label_dir) in self.splits.items():
            class_counts = defaultdict(int)
            
            for label_path in Path(label_dir).glob("*.txt"):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                cls_id = int(parts[0])
                                class_counts[cls_id] += 1
                            except ValueError:
                                pass
            
            distributions[split] = dict(class_counts)
        
        # Check if all classes are represented in each split
        all_classes = set()
        for dist in distributions.values():
            all_classes.update(dist.keys())
        
        missing_classes = {}
        for split, dist in distributions.items():
            missing = all_classes - set(dist.keys())
            if missing:
                missing_classes[split] = list(missing)
        
        passed = len(missing_classes) == 0
        result = {
            "passed": passed,
            "distributions": distributions,
            "missing_classes": missing_classes,
            "total_classes": len(all_classes)
        }
        
        if passed:
            print(f"   ✅ All {len(all_classes)} classes present in all splits")
        else:
            print(f"   ⚠️ Some classes missing in splits")
            for split, missing in missing_classes.items():
                print(f"      [{split}] missing: {[CLASS_NAMES[c] for c in missing[:5]]}")
        
        return result
    
    def save_report(self, output_dir: Path):
        """Save validation report to JSON."""
        report_path = output_dir / "reports" / "data_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        print(f"\n📄 Validation report saved: {report_path}")


# ============================================================================
# PHASE 2: MODEL INFERENCE
# ============================================================================

class ModelTester:
    """Run inference on test data and collect predictions."""
    
    def __init__(self, model_path: Path, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.predictions = []
    
    def load_model(self):
        """Load the YOLO model."""
        from ultralytics import YOLO
        print(f"\n🔄 Loading model: {self.model_name}")
        print(f"   Path: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        print("   ✅ Model loaded successfully")
    
    def run_inference(self, split: str, images_dir: Path, 
                      conf: float = 0.35, iou: float = 0.5) -> List[Dict]:
        """Run inference on a dataset split."""
        print(f"\n🔮 Running inference on {split} split...")
        
        image_paths = list(images_dir.glob("*.[jJ][pP][gG]")) + \
                      list(images_dir.glob("*.[pP][nN][gG]"))
        
        predictions = []
        
        for img_path in tqdm(image_paths, desc=f"   {split}"):
            try:
                # Run prediction
                results = self.model.predict(
                    str(img_path),
                    conf=conf,
                    iou=iou,
                    imgsz=TestConfig.IMAGE_SIZE,
                    verbose=False
                )
                
                # Extract predictions
                result = results[0]
                boxes = result.boxes
                
                pred_data = {
                    "image": img_path.name,
                    "split": split,
                    "detections": [],
                    "num_detections": 0,
                }
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        det = {
                            "class_id": int(box.cls.item()),
                            "class_name": CLASS_NAMES[int(box.cls.item())],
                            "confidence": float(box.conf.item()),
                            "bbox": box.xyxy[0].tolist(),
                            "bbox_center_x": float((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                        }
                        pred_data["detections"].append(det)
                    
                    # Sort by x-coordinate (left to right for VIN)
                    pred_data["detections"].sort(key=lambda x: x["bbox_center_x"])
                    pred_data["num_detections"] = len(pred_data["detections"])
                    pred_data["predicted_vin"] = "".join(
                        d["class_name"] for d in pred_data["detections"]
                    )
                else:
                    pred_data["predicted_vin"] = ""
                
                predictions.append(pred_data)
                
            except Exception as e:
                predictions.append({
                    "image": img_path.name,
                    "split": split,
                    "error": str(e),
                    "num_detections": 0,
                    "predicted_vin": "",
                })
        
        print(f"   ✅ Processed {len(predictions)} images")
        return predictions
    
    def run_full_test(self) -> Dict[str, List[Dict]]:
        """Run inference on all splits."""
        self.load_model()
        
        all_predictions = {}
        
        splits_config = {
            "train": TRAIN_IMAGES,
            "val": VAL_IMAGES,
            "test": TEST_IMAGES,
        }
        
        for split, img_dir in splits_config.items():
            predictions = self.run_inference(
                split, img_dir,
                conf=TestConfig.CONFIDENCE_THRESHOLD,
                iou=TestConfig.IOU_THRESHOLD
            )
            all_predictions[split] = predictions
        
        self.predictions = all_predictions
        return all_predictions


# ============================================================================
# PHASE 3: METRICS CALCULATION
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive metrics from predictions."""
    
    def __init__(self, predictions: Dict[str, List[Dict]], model_name: str):
        self.predictions = predictions
        self.model_name = model_name
        self.metrics = {}
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all metrics."""
        print("\n" + "="*60)
        print("PHASE 3: METRICS CALCULATION")
        print("="*60)
        
        metrics = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "splits": {}
        }
        
        for split, preds in self.predictions.items():
            print(f"\n📊 Calculating metrics for {split} split...")
            
            split_metrics = {
                "total_images": len(preds),
                "images_with_detections": sum(1 for p in preds if p["num_detections"] > 0),
                "total_detections": sum(p["num_detections"] for p in preds),
                "avg_detections_per_image": 0,
                "detection_rate": 0,
                "class_distribution": defaultdict(int),
                "confidence_stats": {},
            }
            
            # Detection rate
            if split_metrics["total_images"] > 0:
                split_metrics["detection_rate"] = (
                    split_metrics["images_with_detections"] / 
                    split_metrics["total_images"] * 100
                )
                split_metrics["avg_detections_per_image"] = (
                    split_metrics["total_detections"] / 
                    split_metrics["total_images"]
                )
            
            # Class distribution and confidence
            all_confidences = []
            for pred in preds:
                for det in pred.get("detections", []):
                    split_metrics["class_distribution"][det["class_name"]] += 1
                    all_confidences.append(det["confidence"])
            
            # Confidence statistics
            if all_confidences:
                split_metrics["confidence_stats"] = {
                    "mean": float(np.mean(all_confidences)),
                    "std": float(np.std(all_confidences)),
                    "min": float(np.min(all_confidences)),
                    "max": float(np.max(all_confidences)),
                    "median": float(np.median(all_confidences)),
                }
            
            # Convert defaultdict to regular dict
            split_metrics["class_distribution"] = dict(split_metrics["class_distribution"])
            
            metrics["splits"][split] = split_metrics
            
            # Print summary
            print(f"   Images: {split_metrics['total_images']}")
            print(f"   Detection Rate: {split_metrics['detection_rate']:.1f}%")
            print(f"   Avg Detections/Image: {split_metrics['avg_detections_per_image']:.1f}")
            if split_metrics["confidence_stats"]:
                print(f"   Mean Confidence: {split_metrics['confidence_stats']['mean']:.3f}")
        
        self.metrics = metrics
        return metrics
    
    def run_official_validation(self) -> Dict:
        """Run official YOLO validation using ultralytics."""
        print("\n🎯 Running official YOLO validation...")
        
        from ultralytics import YOLO
        
        model_path = TestConfig.MODELS.get(self.model_name.replace(" ", "_").lower())
        if model_path is None:
            model_path = list(TestConfig.MODELS.values())[0]
        
        model = YOLO(str(model_path))
        
        official_metrics = {}
        
        for split in ["val", "test"]:
            print(f"\n   Validating on {split} split...")
            try:
                results = model.val(
                    data=str(DATA_YAML),
                    split=split,
                    verbose=False
                )
                
                official_metrics[split] = {
                    "mAP50": float(results.box.map50),
                    "mAP50-95": float(results.box.map),
                    "mAP75": float(results.box.map75),
                    "precision": float(results.box.mp),
                    "recall": float(results.box.mr),
                    "f1": float(2 * results.box.mp * results.box.mr / 
                               (results.box.mp + results.box.mr + 1e-9)),
                }
                
                print(f"      mAP@0.5: {official_metrics[split]['mAP50']:.4f}")
                print(f"      mAP@0.5:0.95: {official_metrics[split]['mAP50-95']:.4f}")
                print(f"      Precision: {official_metrics[split]['precision']:.4f}")
                print(f"      Recall: {official_metrics[split]['recall']:.4f}")
                print(f"      F1: {official_metrics[split]['f1']:.4f}")
                
            except Exception as e:
                print(f"      ⚠️ Validation failed: {e}")
                official_metrics[split] = {"error": str(e)}
        
        self.metrics["official_validation"] = official_metrics
        return official_metrics


# ============================================================================
# PHASE 4: VISUALIZATION
# ============================================================================

class ResultsVisualizer:
    """Create visualizations of test results."""
    
    def __init__(self, metrics: Dict, output_dir: Path):
        self.metrics = metrics
        self.output_dir = output_dir / "plots"
    
    def create_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("PHASE 4: VISUALIZATION")
        print("="*60)
        
        self._plot_detection_rates()
        self._plot_class_distribution()
        self._plot_confidence_distribution()
        self._plot_official_metrics()
        
        print(f"\n✅ All plots saved to: {self.output_dir}")
    
    def _plot_detection_rates(self):
        """Plot detection rates across splits."""
        print("\n📊 Creating detection rate plot...")
        
        splits = list(self.metrics["splits"].keys())
        rates = [self.metrics["splits"][s]["detection_rate"] for s in splits]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(splits, rates, color=['#2ecc71', '#3498db', '#e74c3c'])
        
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title(f'Detection Rate by Split - {self.metrics["model_name"]}')
        ax.set_ylim(0, 100)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "detection_rates.png", dpi=150)
        plt.close()
    
    def _plot_class_distribution(self):
        """Plot class distribution heatmap."""
        print("📊 Creating class distribution heatmap...")
        
        # Aggregate class distributions
        all_classes = set()
        for split_data in self.metrics["splits"].values():
            all_classes.update(split_data["class_distribution"].keys())
        
        all_classes = sorted(all_classes)
        splits = list(self.metrics["splits"].keys())
        
        # Create matrix
        matrix = np.zeros((len(splits), len(all_classes)))
        for i, split in enumerate(splits):
            dist = self.metrics["splits"][split]["class_distribution"]
            for j, cls in enumerate(all_classes):
                matrix[i, j] = dist.get(cls, 0)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=all_classes, yticklabels=splits, ax=ax)
        
        ax.set_title(f'Class Distribution by Split - {self.metrics["model_name"]}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Split')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=150)
        plt.close()
    
    def _plot_confidence_distribution(self):
        """Plot confidence score distributions."""
        print("📊 Creating confidence distribution plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, split in zip(axes, ["train", "val", "test"]):
            stats = self.metrics["splits"][split].get("confidence_stats", {})
            if stats:
                # Create a simple box representation
                data = [stats.get("min", 0), stats.get("median", 0), stats.get("max", 1)]
                ax.boxplot([data], labels=[split])
                ax.axhline(y=stats.get("mean", 0), color='r', linestyle='--', 
                          label=f'Mean: {stats.get("mean", 0):.3f}')
                ax.legend()
            ax.set_ylabel('Confidence')
            ax.set_title(f'{split.capitalize()} Split')
            ax.set_ylim(0, 1)
        
        plt.suptitle(f'Confidence Distribution - {self.metrics["model_name"]}')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_distribution.png", dpi=150)
        plt.close()
    
    def _plot_official_metrics(self):
        """Plot official YOLO validation metrics."""
        print("📊 Creating official metrics comparison plot...")
        
        official = self.metrics.get("official_validation", {})
        if not official:
            return
        
        metrics_names = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        val_values = [official.get("val", {}).get(m, 0) for m in metrics_names]
        test_values = [official.get("test", {}).get(m, 0) for m in metrics_names]
        
        bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Official YOLO Metrics - {self.metrics["model_name"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "official_metrics.png", dpi=150)
        plt.close()


# ============================================================================
# PHASE 5: REPORTING
# ============================================================================

class ReportGenerator:
    """Generate comprehensive test reports."""
    
    def __init__(self, validation_results: Dict, metrics: Dict, 
                 predictions: Dict, output_dir: Path):
        self.validation = validation_results
        self.metrics = metrics
        self.predictions = predictions
        self.output_dir = output_dir
    
    def generate_all_reports(self):
        """Generate all reports."""
        print("\n" + "="*60)
        print("PHASE 5: REPORT GENERATION")
        print("="*60)
        
        self._save_json_report()
        self._save_csv_predictions()
        self._save_summary_report()
        
        print(f"\n✅ All reports saved to: {self.output_dir / 'reports'}")
    
    def _save_json_report(self):
        """Save comprehensive JSON report."""
        print("\n📄 Saving JSON report...")
        
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.metrics.get("model_name", "unknown"),
                "config": {
                    "confidence_threshold": TestConfig.CONFIDENCE_THRESHOLD,
                    "iou_threshold": TestConfig.IOU_THRESHOLD,
                    "image_size": TestConfig.IMAGE_SIZE,
                }
            },
            "data_validation": self.validation,
            "metrics": self.metrics,
        }
        
        report_path = self.output_dir / "reports" / "full_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   Saved: {report_path}")
    
    def _save_csv_predictions(self):
        """Save predictions to CSV."""
        print("📄 Saving predictions CSV...")
        
        rows = []
        for split, preds in self.predictions.items():
            for pred in preds:
                rows.append({
                    "split": split,
                    "image": pred["image"],
                    "num_detections": pred["num_detections"],
                    "predicted_vin": pred.get("predicted_vin", ""),
                    "error": pred.get("error", ""),
                })
        
        csv_path = self.output_dir / "predictions" / "all_predictions.csv"
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"   Saved: {csv_path}")
    
    def _save_summary_report(self):
        """Save human-readable summary report."""
        print("📄 Saving summary report...")
        
        report_lines = [
            "="*70,
            "JLR VIN YOLO MODEL - TEST RESULTS SUMMARY",
            "="*70,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.metrics.get('model_name', 'unknown')}",
            "",
            "-"*70,
            "DATA VALIDATION",
            "-"*70,
        ]
        
        if self.validation.get("all_passed"):
            report_lines.append("✅ All validation checks PASSED")
        else:
            report_lines.append("⚠️ Some validation checks FAILED")
        
        for check_name, check_result in self.validation.get("checks", {}).items():
            status = "✅" if check_result.get("passed") else "❌"
            report_lines.append(f"   {status} {check_name}")
        
        report_lines.extend([
            "",
            "-"*70,
            "DETECTION METRICS BY SPLIT",
            "-"*70,
        ])
        
        for split, data in self.metrics.get("splits", {}).items():
            report_lines.extend([
                f"\n{split.upper()}:",
                f"   Images: {data['total_images']}",
                f"   Detection Rate: {data['detection_rate']:.1f}%",
                f"   Total Detections: {data['total_detections']}",
                f"   Avg Detections/Image: {data['avg_detections_per_image']:.2f}",
            ])
            if data.get("confidence_stats"):
                report_lines.append(
                    f"   Mean Confidence: {data['confidence_stats']['mean']:.3f}"
                )
        
        # Official metrics
        official = self.metrics.get("official_validation", {})
        if official:
            report_lines.extend([
                "",
                "-"*70,
                "OFFICIAL YOLO VALIDATION METRICS",
                "-"*70,
            ])
            
            for split in ["val", "test"]:
                if split in official and "error" not in official[split]:
                    m = official[split]
                    report_lines.extend([
                        f"\n{split.upper()}:",
                        f"   mAP@0.5:      {m['mAP50']:.4f}",
                        f"   mAP@0.5:0.95: {m['mAP50-95']:.4f}",
                        f"   Precision:    {m['precision']:.4f}",
                        f"   Recall:       {m['recall']:.4f}",
                        f"   F1 Score:     {m['f1']:.4f}",
                    ])
        
        report_lines.extend([
            "",
            "="*70,
            "END OF REPORT",
            "="*70,
        ])
        
        report_path = self.output_dir / "reports" / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        # Also print to console
        print("\n" + "\n".join(report_lines))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("JLR VIN YOLO MODEL - ENTERPRISE TESTING PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    TestConfig.setup()
    
    # Select model to test
    print("\n📋 Available models:")
    for i, (name, path) in enumerate(TestConfig.MODELS.items(), 1):
        exists = "✅" if Path(path).exists() else "❌"
        print(f"   {i}. {name}: {path} {exists}")
    
    # Use the first available model
    model_name = None
    model_path = None
    for name, path in TestConfig.MODELS.items():
        if Path(path).exists():
            model_name = name
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found! Please train a model first.")
        return
    
    print(f"\n🎯 Using model: {model_name}")
    
    # Phase 1: Data Validation
    if TestConfig.RUN_DATA_VALIDATION:
        validator = DataValidator()
        validation_results = validator.run_all_checks()
        validator.save_report(TestConfig.RESULTS_DIR)
        
        if not validation_results["all_passed"]:
            print("\n⚠️ WARNING: Data validation issues detected!")
            proceed = input("Continue with testing? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Testing aborted.")
                return
    else:
        validation_results = {}
    
    # Phase 2: Model Inference
    if TestConfig.RUN_INFERENCE_TEST:
        tester = ModelTester(Path(model_path), model_name)
        predictions = tester.run_full_test()
    else:
        predictions = {}
    
    # Phase 3: Metrics Calculation
    if TestConfig.RUN_METRICS_ANALYSIS and predictions:
        calculator = MetricsCalculator(predictions, model_name)
        metrics = calculator.calculate_all_metrics()
        calculator.run_official_validation()
    else:
        metrics = {}
    
    # Phase 4: Visualization
    if TestConfig.RUN_VISUALIZATION and metrics:
        visualizer = ResultsVisualizer(metrics, TestConfig.RESULTS_DIR)
        visualizer.create_all_plots()
    
    # Phase 5: Report Generation
    if TestConfig.EXPORT_DETAILED_RESULTS:
        reporter = ReportGenerator(
            validation_results, metrics, predictions, TestConfig.RESULTS_DIR
        )
        reporter.generate_all_reports()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"Results saved to: {TestConfig.RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
