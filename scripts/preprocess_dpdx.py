"""
This script prepares microscopy images for training a CycleGAN model for
Entamoeba cysts image deblurring (DPDx official images ↔ personal 4K laboratory images).

It performs the following operations:

1. Loads images from two domains:
   - Domain A: Off-focus / blurry 4K images (entamoeba_coli + entamoeba_histo)
     - Center-crop to 1000×1000
     - Resize to 256×256
   - Domain B: Clear / high-quality DPDx reference images
     - Directly resize to 256×256 (no cropping)

2. Preprocessing applied to all images:
   - 
   - Save in CycleGAN dataset structure:
        datasets/dpdx_cysts/
            trainA/
            trainB/
            testA/
            testB/

3. Splits each domain into:
   - 90% training
   - 10% testing

This script should be placed inside the cloned repo:
    pytorch-CycleGAN-and-pix2pix/scripts/

Run it from the project root:

    conda activate pytorch-img2img
    python scripts/preprocess_dpdx.py


"""

import os
from pathlib import Path
import random
from PIL import Image, ImageOps

# -----------------------------
# CONFIGURATION
# -----------------------------

# Input folders
DPDX_DIR = Path(r"D:\2025_PROJECT\RawData_CycleGAN")  # clear reference
COLI_DIR = Path(r"D:\2025_PROJECT\PyTorchTesting\data_raw\entamoeba_coli")
HISTO_DIR = Path(r"D:\2025_PROJECT\PyTorchTesting\data_raw\entamoeba_histo")

# Output CycleGAN dataset folder
# OUTPUT_ROOT = Path("datasets/dpdx_cysts")  # no enough space in local pc
OUTPUT_ROOT = Path(r"D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts") # save to external drive D
# Crop & resize parameters
CROP_SIZE_A = 1000 # Domain A only
FINAL_SIZE = 256 # both domains

# Train/test split ratio
TRAIN_RATIO = 0.9


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------


def load_image(path):
    """Loads an image safely and converts to RGB if the image is not already in RGB mode."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"[ERROR] Cannot load image {path}: {e}")
        return None

def validate_image_size(img, min_size=200):
    """Ensure images meet minimum size requirements"""
    w, h = img.size
    if w < min_size or h < min_size:
        raise ValueError(f"Image too small: {w}x{h}")
    return True

def center_crop(img, target):
    """
    Attempts to center-crop the image to a square of side `target`.
   
    Returns:
        PIL.Image
    """
    w, h = img.size
    if w < target or h < target:
        print(f"[WARNING] Image {w}x{h} smaller than crop size {target}")
        # Alternative: resize instead of crop
        return img.resize((target, target), Image.BICUBIC)

    # Normal center-crop
    left = (w - target) // 2
    top = (h - target) // 2
    return img.crop((left, top, left + target, top + target))



def resize(img, size=256):
    """Resize image to CycleGAN training size."""
    return img.resize((size, size), Image.BICUBIC)


def save_image(img, path):
    """Saves the image safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def get_all_images(directory):
    """Returns a list of all supported image files in the directory."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


# Consider adding data augmentation for training, apply only to training images in process_domain function

def augment_image(img):
    """Optional data augmentation for training images"""
    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    # Random rotation (small angles)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle)
    return img




# -----------------------------
# PROCESSING PIPELINE
# -----------------------------

def process_domain(images, out_train, out_test, domain_name="Domain", crop=False, crop_size=1000):
    """
    Processes a list of image paths and saves them into train/test folders.

    Args:
        images (list[Path]): List of image paths
        out_train (Path): Output folder for training
        out_test (Path): Output folder for testing
        domain_name (str): Name for logging
        crop (bool): Whether to center-crop
        crop_size (int): Crop size (for domain A)
    """
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_idx]
    test_images = images[split_idx:]

    
    print(f"\n[{domain_name}] Train: {len(train_images)}, Test: {len(test_images)}")

    for subset, out_dir in [(train_images, out_train), (test_images, out_test)]:
        
        for i, img_path in enumerate(subset):
            if i % 100 == 0:  # Progress tracking
                print(f"  Processed {i}/{len(subset)} images...")
                
            img = load_image(img_path)
            if img is None:
                continue
                
            try:
                validate_image_size(img)
                if crop:
                    img = center_crop(img, crop_size)
                img = resize(img, FINAL_SIZE)
                
                # Generate unique filename to avoid overwrites
                unique_name = f"{img_path.stem}_{i:04d}{img_path.suffix}"
                out_path = out_dir / unique_name
                save_image(img, out_path)
                
            except Exception as e:
                print(f"[ERROR] Processing {img_path}: {e}")
                continue


# -----------------------------
# Dataset Statistics & Verification
# -----------------------------

def verify_dataset():
    """Verify the final dataset structure and statistics"""
    domains = ["trainA", "trainB", "testA", "testB"]
    
    for domain in domains:
        domain_path = OUTPUT_ROOT / domain
        images = list(domain_path.glob("*.*"))
        print(f"{domain}: {len(images)} images")
        
        # Check image sizes
        if images:
            sample = Image.open(images[0])
            print(f"  Sample size: {sample.size}")

# Call this at the end of main()


# -----------------------------
# MAIN SCRIPT
# -----------------------------

def main():

    print("--------------------------------------------------")
    print("CycleGAN Data Preparation for Entamoeba Deblurring")
    print("--------------------------------------------------")

    # Domain B (DPDx clear images)
    print("\n[Domain B] Loading DPDx clear reference images...")
    dpdx_images = get_all_images(DPDX_DIR)
    print(f"  Found {len(dpdx_images)} images.")

    # Domain A (blurred)
    print("\n[Domain A] Loading 4K off-focus lab images...")
    coli_images = get_all_images(COLI_DIR)
    histo_images = get_all_images(HISTO_DIR)
    domain_a_images = coli_images + histo_images
    print(f"  Found {len(domain_a_images)} images.")

    # Output structure
    trainA = OUTPUT_ROOT / "trainA"
    trainB = OUTPUT_ROOT / "trainB"
    testA = OUTPUT_ROOT / "testA"
    testB = OUTPUT_ROOT / "testB"

    # Process both domains
    print("\nProcessing Domain A (blurred)...")
    process_domain(domain_a_images, trainA, testA, domain_name="Domain A (blurred)", crop=True, crop_size=CROP_SIZE_A)

    print("\nProcessing Domain B (clear DPDx)...")
    process_domain(dpdx_images, trainB, testB, domain_name="Domain B (DPDx)", crop=False)

    print("\nDone! Dataset saved to:")
    print(f"  {OUTPUT_ROOT.resolve()}")

    # Call verification at the end
    verify_dataset()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run - no files will be saved")
        # Modify functions to skip saving
    
    
    main()
