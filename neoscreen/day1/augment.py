"""
Day 1 — Task 4 (Afternoon, 2 hrs)
Albumentations augmentation pipeline.
Called internally by ml/train.py; also usable standalone for previewing.

Augmentations applied (training only):
  - Rotation ±15°
  - Brightness / contrast variation ±30%
  - Horizontal flip
  - White balance shift (hue/saturation)
  - CLAHE (additional contrast enhancement)
  - Gaussian noise

Usage:
    # Preview augmentations on a single image (saves grid to day1/aug_preview.jpg)
    python day1/augment.py --preview dataset/High/0001.jpg

    # Expand a small dataset 3× by saving augmented copies
    python day1/augment.py --expand --data_dir ./dataset --multiplier 3
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("WARNING: albumentations not installed. Using basic OpenCV augmentation fallback.")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def get_train_transform():
    """Full augmentation pipeline used during training."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.20,
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=0.6,
            ),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
    else:
        return None  # fallback handled in apply_augmentation()


def apply_augmentation(image: np.ndarray, transform=None) -> np.ndarray:
    """
    Apply augmentation to a single RGB uint8 image (H×W×3).
    Returns augmented RGB uint8 image of the same size.
    """
    if transform is None:
        transform = get_train_transform()

    if HAS_ALBUMENTATIONS and transform is not None:
        result = transform(image=image)
        return result["image"]
    else:
        # OpenCV-only fallback
        return _opencv_augment(image)


def _opencv_augment(img: np.ndarray) -> np.ndarray:
    """Minimal OpenCV augmentation when Albumentations is unavailable."""
    h, w = img.shape[:2]
    rng = np.random.default_rng()

    # Random rotation ±15°
    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Random horizontal flip
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    # Brightness / contrast
    alpha = rng.uniform(0.70, 1.30)   # contrast
    beta = rng.uniform(-30, 30)        # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


# ── Preview ───────────────────────────────────────────────────────────────────

def preview(image_path: str, n_cols: int = 4, n_rows: int = 2):
    """Generate a grid showing original + 7 augmented versions."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read {image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))

    transform = get_train_transform()
    variants = [img_rgb]  # first cell = original
    for _ in range(n_cols * n_rows - 1):
        variants.append(apply_augmentation(img_rgb.copy(), transform))

    # Build grid
    rows = []
    for r in range(n_rows):
        row_imgs = variants[r * n_cols:(r + 1) * n_cols]
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    out_path = "day1/aug_preview.jpg"
    os.makedirs("day1", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Augmentation preview saved → {out_path}")
    print("  Top-left = original. Remaining cells = augmented variants.")


# ── Expand dataset ────────────────────────────────────────────────────────────

def expand_dataset(data_dir: str, multiplier: int = 3):
    """
    Save multiplier-1 augmented copies of every image in the dataset.
    Adds aug_1_, aug_2_, ... prefix to filenames.
    """
    CLASSES = ["Low", "Medium", "High"]
    transform = get_train_transform()
    total_added = 0

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Skipping {cls} — folder not found")
            continue

        originals = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))
                     and not f.startswith("aug_")]

        print(f"\n  {cls}: {len(originals)} originals → generating {len(originals) * (multiplier - 1)} copies...")

        for fname in originals:
            src_path = os.path.join(cls_dir, fname)
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)

            for i in range(1, multiplier):
                aug_img = apply_augmentation(img_rgb.copy(), transform)
                out_name = f"aug_{i}_{fname}"
                out_path = os.path.join(cls_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                total_added += 1

    print(f"\n  Done. Added {total_added} augmented images.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    # preview subcommand
    p_prev = subparsers.add_parser("--preview", help="Generate augmentation preview grid")
    p_prev.add_argument("image", help="Source image path")

    # expand subcommand
    p_exp = subparsers.add_parser("--expand", help="Expand dataset with augmented copies")
    p_exp.add_argument("--data_dir", default="./dataset")
    p_exp.add_argument("--multiplier", type=int, default=3)

    # Handle --preview / --expand as top-level flags (simpler CLI)
    args, remaining = parser.parse_known_args()

    import sys as _sys
    argv = _sys.argv[1:]

    if "--preview" in argv:
        idx = argv.index("--preview")
        img_path = argv[idx + 1] if idx + 1 < len(argv) else None
        if not img_path:
            print("Usage: python day1/augment.py --preview path/to/image.jpg")
            _sys.exit(1)
        preview(img_path)

    elif "--expand" in argv:
        data_dir = "./dataset"
        multiplier = 3
        if "--data_dir" in argv:
            data_dir = argv[argv.index("--data_dir") + 1]
        if "--multiplier" in argv:
            multiplier = int(argv[argv.index("--multiplier") + 1])
        expand_dataset(data_dir, multiplier)

    else:
        print("Usage:")
        print("  python day1/augment.py --preview dataset/High/0001.jpg")
        print("  python day1/augment.py --expand --data_dir ./dataset --multiplier 3")
