import os
import glob
import numpy as np
from PIL import Image

# ROOT FOLDER
DATA_ROOT = "/Users/katerinaskrika/Desktop/cool_project/data"
DATASET_NAME = "ISIC2019"
NUM_CLASSES = 2   # classes for random masks

def generate_masks_for_split(SPLIT):
    print(f"\n=== Generating masks for split: {SPLIT} ===")

    # Image directory (same structure as before)
    image_dir = os.path.join(
        DATA_ROOT,
        DATASET_NAME,
        SPLIT,
        "images",
        "ISIC_2019_Training_Input"
    )

    if not os.path.isdir(image_dir):
        print(f"[{SPLIT}] No image directory found → skipping")
        return

    # Mask directory
    mask_dir = os.path.join(
        DATA_ROOT,
        DATASET_NAME,
        SPLIT,
        "masks"
    )
    os.makedirs(mask_dir, exist_ok=True)

    # Collect all .jpg images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    print(f"[{SPLIT}] Found {len(image_paths)} images")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Create random mask
        mask_np = np.random.randint(0, NUM_CLASSES, size=(h, w), dtype=np.uint8)
        mask = Image.fromarray(mask_np, mode="L")

        # Save mask with same basename
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + ".png")
        mask.save(mask_path)

    print(f"[{SPLIT}] Saved {len(image_paths)} random masks in {mask_dir}")


# -----------------------------------------------------
# Run for train + test
# -----------------------------------------------------
generate_masks_for_split("train")
generate_masks_for_split("test")
