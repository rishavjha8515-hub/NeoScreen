# Day 1 — Saturday: Core ML Pipeline

Goal: Working Python ML pipeline + trained .tflite model by end of day.

---

## Morning — Task 1 (3 hrs): Dataset Setup

```bash
# Download Bilicam + NIH sample images and organise into Low/Medium/High folders
python day1/dataset_setup.py
```

Expected output:
```
dataset/
  Low/      ← ~600 images
  Medium/   ← ~600 images
  High/     ← ~600 images
```

If you have access to the full Bilicam dataset (UW), copy it manually:
```bash
cp -r /path/to/bilicam/* dataset/
```

---

## Morning — Task 2 (1 hr): Sclera Detection

Already built at `ml/sclera_detection.py`. Test it now:

```bash
# Quick smoke test — generates a synthetic eye image and runs detection
python day1/test_sclera.py
```

If it prints `✓ Sclera detected`, the OpenCV pipeline is working.

---

## Afternoon — Task 3 (2 hrs): Load MobileNetV3 + Build 3-Class Head

Open the Colab notebook:

```
NeoScreen_Training.ipynb  →  Runtime → Change runtime type → T4 GPU
```

Or run locally (slower):
```bash
python day1/build_model.py   # prints model summary and verifies architecture
```

---

## Afternoon — Task 4 (2 hrs): Train + Augmentation

```bash
# Run full training with Albumentations augmentation
python ml/train.py --data_dir ./dataset --epochs 25 --output neoscreen_v1.tflite
```

The augmentation pipeline is in `day1/augment.py` — train.py calls it internally.
To preview what augmented images look like:

```bash
python day1/augment.py --preview dataset/High/0001.jpg
```

---

## Evening — Task 5 (1 hr): Export + Smoke-Test on 10 Images

```bash
# Verify model exported correctly, runs in <500ms, size <5MB
python day1/test_10_images.py --model neoscreen_v1.tflite --data_dir ./dataset
```

---

## Evening — Task 6 (1 hr): Log Misclassifications

```bash
# Run full evaluation + write misclassification report
python day1/misclassification_log.py --model neoscreen_v1.tflite --data_dir ./dataset
# Output: day1/reports/misclassification_report.txt
```

---

## End of Day 1 Checklist

- [ ] `dataset/` populated (Low / Medium / High subdirs)
- [ ] `python day1/test_sclera.py` passes
- [ ] `neoscreen_v1.tflite` exists and is < 5 MB
- [ ] `python day1/test_10_images.py` prints results for 10 images
- [ ] `day1/reports/misclassification_report.txt` written
- [ ] Note sensitivity score — target ≥ 95% on High Risk

If sensitivity < 95%: lower `THRESHOLD_HIGH` in `ml/inference.py` (try 0.30) and re-evaluate.
