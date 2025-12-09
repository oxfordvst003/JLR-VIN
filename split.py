import os
import cv2
import easyocr

# Import centralized config
try:
    from config import DATA_ROOT
    dataset_root = str(DATA_ROOT)
except ImportError:
    dataset_root = r"C:\JLR_VIN_2549\Data"  # fallback

splits = ["train", "val", "test"]

# Initialize EasyOCR (English + numbers)
reader = easyocr.Reader(['en'], gpu=True)  # set gpu=False if no CUDA

# Loop over dataset splits
for split in splits:
    img_folder = os.path.join(dataset_root, split, "images")
    label_folder = os.path.join(dataset_root, split, "labels")
    os.makedirs(label_folder, exist_ok=True)

    for img_file in os.listdir(img_folder):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_folder, img_file)
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")

        # Run OCR
        results = reader.readtext(img_path, detail=1)

        # Load image to get dimensions
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, "w") as f:
            for (bbox, text, prob) in results:
                # bbox has 4 corner points of box
                (x_min, y_min) = bbox[0]
                (x_max, y_max) = bbox[2]

                # Normalize coordinates for YOLO format
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                box_w = (x_max - x_min) / w
                box_h = (y_max - y_min) / h

                # Character-level split (each char becomes one object)
                for char in text:
                    # Get class ID (A=0, B=1, … Z=25, 0=26, … 9=35)
                    if char.isalpha():
                        cls = ord(char.upper()) - 65  # A=0
                    elif char.isdigit():
                        cls = 26 + int(char)         # 0=26
                    else:
                        continue  # ignore symbols like *

                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        print(f" Processed {img_file}, labels saved at {label_path}")
