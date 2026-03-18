import cv2
import numpy as np
from typing import Optional


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Low-light enhancement via CLAHE on the L channel (LAB colour space)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_white_balance(img: np.ndarray) -> np.ndarray:
    """Automatic white balance using grey-world assumption in LAB space."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] -= (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    result[:, :, 2] -= (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def detect_sclera(image_path: str) -> Optional[np.ndarray]:
    """
    Full pipeline: load → CLAHE → white balance → HSV mask → contour → crop → resize.

    Returns:
        224×224 BGR numpy array if sclera found, else None (prompt user to retry).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = apply_clahe(img)
    img = apply_white_balance(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 20 or h < 20:
        return None  # noise — not a real sclera region

    sclera_roi = img[y:y + h, x:x + w]
    return cv2.resize(sclera_roi, (224, 224))


def preprocess_for_inference(sclera_224: np.ndarray) -> np.ndarray:
    """Convert BGR 224×224 → normalised RGB float32 with batch dim."""
    rgb = cv2.cvtColor(sclera_224, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalised, axis=0)  # (1, 224, 224, 3)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    roi = detect_sclera(path)
    if roi is None:
        print("Sclera not detected. Please retake photo.")
    else:
        cv2.imwrite("sclera_roi.jpg", roi)
        print("Sclera detected → sclera_roi.jpg")