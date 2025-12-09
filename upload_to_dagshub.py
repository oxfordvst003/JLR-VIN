"""
Script to upload Data folder to DagsHub storage
"""
import os
from dagshub.upload import Repo

# Initialize repo
repo = Repo("oxfordvst003", "JLR-VIN")

# Define folders to upload
data_folders = [
    ("Data/train/images", "Data/train/images"),
    ("Data/train/labels", "Data/train/labels"),
    ("Data/val/images", "Data/val/images"),
    ("Data/val/labels", "Data/val/labels"),
    ("Data/test/images", "Data/test/images"),
    ("Data/test/labels", "Data/test/labels"),
]

# Upload each folder
for local_path, remote_path in data_folders:
    if os.path.exists(local_path):
        print(f"Uploading {local_path} -> {remote_path}")
        try:
            repo.upload(
                local_path=local_path,
                remote_path=remote_path,
                commit_message=f"Upload {local_path}",
                versioning="dvc"  # Use DVC for large files
            )
            print(f"  ✓ Uploaded {local_path}")
        except Exception as e:
            print(f"  ✗ Failed to upload {local_path}: {e}")
    else:
        print(f"  ⚠ Folder not found: {local_path}")

print("\nUpload complete!")
