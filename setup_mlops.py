"""
================================================================================
MLOps Setup Script
================================================================================
Initialize the MLOps infrastructure for YOLO training.
Run this script after installing dependencies to set up:
- DagsHub repository connection
- MLflow tracking
- DVC data versioning
- Model registry
================================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required = [
        ('mlflow', 'mlflow'),
        ('dagshub', 'dagshub'),
        ('dvc', 'dvc'),
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('psutil', 'psutil'),
    ]
    
    missing = []
    for package, import_name in required:
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def init_dvc():
    """Initialize DVC for data versioning."""
    print_header("Initializing DVC")
    
    dvc_dir = PROJECT_ROOT / ".dvc"
    
    if dvc_dir.exists():
        print("  DVC already initialized")
    else:
        try:
            subprocess.run(['dvc', 'init'], cwd=PROJECT_ROOT, check=True)
            print("  ✓ DVC initialized")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ DVC init failed: {e}")
            return False
        except FileNotFoundError:
            print("  ✗ DVC not found. Install with: pip install dvc")
            return False
    
    # Track data folder
    data_dvc = PROJECT_ROOT / "Data.dvc"
    if not data_dvc.exists():
        try:
            subprocess.run(['dvc', 'add', 'Data'], cwd=PROJECT_ROOT, check=True)
            print("  ✓ Data folder tracked by DVC")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Could not add Data folder: {e}")
    else:
        print("  Data folder already tracked")
    
    return True


def init_mlops_directories():
    """Create MLOps directory structure."""
    print_header("Creating MLOps Directories")
    
    directories = [
        PROJECT_ROOT / "mlops" / "outputs",
        PROJECT_ROOT / "mlops" / "registry",
        PROJECT_ROOT / "mlops" / "lineage",
        PROJECT_ROOT / "mlops" / "reports",
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path.relative_to(PROJECT_ROOT)}")
    
    return True


def validate_data():
    """Validate data structure."""
    print_header("Validating Data Structure")
    
    from config import DATA_ROOT
    
    splits = ['train', 'val', 'test']
    all_valid = True
    
    for split in splits:
        img_dir = DATA_ROOT / split / "images"
        label_dir = DATA_ROOT / split / "labels"
        
        if img_dir.exists():
            images = list(img_dir.glob('*.[jJ][pP][gG]')) + list(img_dir.glob('*.[pP][nN][gG]'))
            labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []
            
            status = "✓" if len(images) > 0 else "✗"
            match_status = "✓" if len(images) == len(labels) else "⚠"
            
            print(f"  {status} {split}: {len(images)} images, {len(labels)} labels {match_status}")
            
            if len(images) == 0:
                all_valid = False
        else:
            print(f"  ✗ {split}: Directory not found")
            all_valid = False
    
    return all_valid


def setup_dagshub():
    """Configure DagsHub connection."""
    print_header("DagsHub Configuration")
    
    print("  To connect to DagsHub:")
    print("  1. Create a repository at https://dagshub.com")
    print("  2. Update mlops/mlops_config.py with your:")
    print("     - DagsHub username (owner)")
    print("     - Repository name")
    print("  3. Set your DagsHub token:")
    print("     export DAGSHUB_TOKEN=your_token")
    print("  4. Or use: dagshub login")
    print()
    
    # Check if token is set
    token = os.environ.get('DAGSHUB_TOKEN')
    if token:
        print("  ✓ DAGSHUB_TOKEN is set")
    else:
        print("  ⚠ DAGSHUB_TOKEN not set (optional for local development)")
    
    return True


def test_mlops_import():
    """Test that MLOps module can be imported."""
    print_header("Testing MLOps Module")
    
    try:
        from mlops import (
            MLOPS_CONFIG,
            mlflow_tracker,
            data_versioner,
            model_registry,
            MLOpsPipeline
        )
        print("  ✓ MLOps module imports successful")
        
        # Test data versioner
        stats = data_versioner.get_data_stats()
        print(f"  ✓ Data hash: {stats['data_hash']}")
        print(f"  ✓ Total images: {stats['total_images']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! Next Steps")
    
    print("""
  1. CONFIGURE DAGSHUB (Optional but Recommended):
     - Create repo at https://dagshub.com
     - Update mlops/mlops_config.py with your credentials
     - Run: dagshub login
  
  2. RUN TRAINING WITH MLOPS:
     python -m mlops.pipeline train --epochs 50 --batch 16
  
  3. RUN EVALUATION:
     python -m mlops.pipeline evaluate --split test
  
  4. CHECK STATUS:
     python -m mlops.pipeline status
  
  5. RUN FULL DVC PIPELINE:
     dvc repro
  
  6. VIEW EXPERIMENTS:
     mlflow ui --port 5000
     # Then open http://localhost:5000
  
  For more information, see mlops/README.md
""")


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("  JLR VIN YOLO - MLOps Setup")
    print("=" * 60)
    
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Initializing directories", init_mlops_directories),
        ("Validating data", validate_data),
        ("Initializing DVC", init_dvc),
        ("Configuring DagsHub", setup_dagshub),
        ("Testing MLOps module", test_mlops_import),
    ]
    
    results = []
    for name, func in steps:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Setup Summary")
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n  ✓ All setup steps completed successfully!")
        print_next_steps()
    else:
        print("\n  ⚠ Some setup steps had issues. Please review above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
