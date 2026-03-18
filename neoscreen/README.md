# NeoScreen 🔍

**Newborn Jaundice Detection via Smartphone Camera**  
Free · Offline · Android 8+ · Hindi & English  
IdeaSprint Lab2Launch Finals — IISc Bangalore 2026

---

## What It Does

NeoScreen detects newborn jaundice risk by analysing the colour of a baby's eye sclera. Point any Android smartphone camera at the baby's eye — the app returns a **Low / Medium / High** risk classification in Hindi or English in under 30 seconds. On High Risk, it automatically sends an SMS to the nearest PHC before the parent even arrives.

The entire 4 MB CNN model runs **on-device** — no internet, no cloud, no cost per screening.

---

## The Problem

| Stat | Source |
|------|--------|
| 60% of Indian newborns develop jaundice within 72 hours | WHO |
| 80% of rural births happen at home — zero monitoring | HMIS |
| 1,00,000+ kernicterus cases annually from late detection | UNICEF India |
| Blood test costs ₹500–2,000, requires a lab, unavailable in 6,50,000 villages | NITI Aayog |

**NeoScreen cost per screening: ₹0**

---

## Architecture Overview

```
Smartphone Camera
       │
       ▼
┌──────────────────┐
│  CLAHE + White   │  OpenCV — low-light & colour correction
│  Balance         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Sclera ROI      │  HSV masking → contour detection → crop
│  Detection       │
└────────┬─────────┘
         │  224×224 image
         ▼
┌──────────────────┐
│  MobileNetV3     │  4 MB TF Lite — fully offline
│  TF Lite         │  Pretrained ImageNet + fine-tuned sclera
└────────┬─────────┘
         │  [P(Low), P(Medium), P(High)]
         ▼
┌──────────────────┐
│  Classification  │  Safety-first thresholds
│  + Language      │  Hindi / English output
└────────┬─────────┘
         │  HIGH only
         ▼
┌──────────────────┐
│  PHC SMS Alert   │  GeoPy Haversine → nearest PHC → Twilio
│  (Twilio)        │  No photo. No PII stored.
└──────────────────┘
```

---

## Repository Structure

```
neoscreen/
├── ml/
│   ├── sclera_detection.py   # CLAHE + white balance + HSV ROI crop
│   ├── inference.py          # TF Lite inference + threshold classification
│   ├── referral.py           # GeoPy nearest PHC + Twilio SMS
│   ├── train.py              # MobileNetV3 transfer learning training
│   └── evaluate.py           # Sensitivity, Kappa, AUC-ROC evaluation
├── flutter/
│   ├── pubspec.yaml
│   └── lib/
│       ├── main.dart
│       ├── screens/
│       │   ├── main_screen.dart    # Camera → capture → analyse
│       │   └── result_screen.dart  # Risk display + confidence bars
│       ├── services/
│       │   ├── inference_service.dart   # tflite_flutter wrapper
│       │   ├── sclera_detector.dart     # Native Dart sclera detection
│       │   └── phc_referral_service.dart # SQLite PHC lookup + Twilio SMS
│       ├── models/
│       │   └── risk_result.dart
│       └── utils/
│           └── locale_utils.dart
├── scripts/
│   └── seed_phc_db.py        # Seed 100 Maharashtra PHC coordinates
├── NeoScreen_Training.ipynb  # Google Colab training notebook (free T4)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/neoscreen.git
cd neoscreen
```

### 2. Python ML pipeline

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM
python scripts/seed_phc_db.py # creates data/phc_maharashtra.db

# Test sclera detection on a sample image
python ml/sclera_detection.py path/to/baby_eye.jpg

# Run end-to-end inference (requires neoscreen_v1.tflite)
python ml/inference.py path/to/baby_eye.jpg
```

### 3. Train the model (Google Colab recommended)

Open `NeoScreen_Training.ipynb` in Google Colab.  
Runtime → Change runtime type → **T4 GPU**.  
Run all cells. Download `neoscreen_v1.tflite` when done.

**Or train locally:**
```bash
python ml/train.py --data_dir ./dataset --epochs 25 --output neoscreen_v1.tflite
python ml/evaluate.py --model neoscreen_v1.tflite --test_dir ./dataset/test
```

### 4. Flutter app

```bash
# Place trained model:
cp neoscreen_v1.tflite flutter/assets/models/

# Seed PHC db for app:
cp data/phc_maharashtra.db flutter/assets/data/

cd flutter
flutter pub get
flutter run                   # connect Android 8+ device or emulator
flutter build apk --release   # build release APK
```

---

## Classification Thresholds

| Condition | Risk | Message (Hindi) | Message (English) |
|-----------|------|-----------------|-------------------|
| P(High) ≥ 0.35 | **HIGH** | Shishu ko turant PHC le jaayein | Take baby to PHC immediately |
| P(Low) ≥ 0.60 | **LOW** | Koi khatra nahi | No immediate risk |
| Otherwise | **MEDIUM** | Kal dobara check karein | Recheck tomorrow |

The High threshold is **deliberately low at 0.35** — uncertain cases always escalate. Over-referral is safer than under-detection.

---

## Model Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sensitivity (High Risk recall) | ≥ 95% | Missing High Risk = kernicterus |
| Specificity | 60–70% | Accepted tradeoff for safety |
| Cohen's Kappa | > 0.80 | Agreement with clinical TSB blood test |
| AUC-ROC | > 0.92 | Discrimination across thresholds |
| Inference time | < 500 ms | On Moto E13 (₹6,000 device) |
| Model size | < 5 MB | Offline APK, Android 8+ |

---

## Dataset

| Dataset | Source | Use |
|---------|--------|-----|
| Bilicam | University of Washington (2016) | Primary training — 2,000+ neonatal eye images with clinical bilirubin labels |
| NIH Neonatal Library | National Institutes of Health | Supplementary — diverse skin tones, lighting |
| Synthetic augmentation | OpenCV + Albumentations | 3× training data — rotation ±15°, brightness ±30%, white balance variation |
| IISc Medical Network | IISc clinical partners | Ground truth validation — clinical TSB vs NeoScreen, target n=200+ |

---

## Tech Stack

| Layer | Tool | Version |
|-------|------|---------|
| Vision | OpenCV | 4.8+ |
| ML | MobileNetV3 TF Lite | 4 MB |
| Training | TensorFlow | 2.13+ |
| Augmentation | Albumentations | 1.3+ |
| Compute | Google Colab T4 | Free |
| App | Flutter + Dart | 3.x |
| On-device ML | tflite_flutter | 0.10.4 |
| Database | SQLite / sqflite | 2.3 |
| SMS | Twilio | Free tier (1,000 SMS/month) |
| Location | GeoPy / Geolocator | Latest |
| Target device | Moto E13 | ₹6,000 — ASHA worker standard |

---

## Privacy

- No photo is ever transmitted off-device
- No personally identifiable information is stored
- SMS to PHC contains only: baby age in hours, ASHA worker ID, GPS coordinates, risk score
- All inference runs locally on the device

---

## Team

**Rishav Anand Kumar Jha** — Technical Lead  
**Ram Barabde** — Research & Strategy Lead  

IdeaSprint Lab2Launch Finals · IISc Bangalore 2026

---

## License

MIT — free to use, adapt, and deploy in public health contexts.
