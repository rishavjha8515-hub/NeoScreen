# Day 3 — Monday: Model Accuracy + Real Data

Goal: Push High Risk sensitivity above 95% using real images and threshold tuning.

---

## Task 1 (2 hrs): Replace synthetic data with real images
```
python day3/download_real_data.py
```
Downloads 50 public NIH neonatal eye images per class.
Organises into dataset/Low, Medium, High.

---

## Task 2 (2 hrs): Retrain on real + synthetic combined
```
python ml/train.py --data_dir ./dataset --epochs 30 --output neoscreen_v1.tflite
```

---

## Task 3 (1 hr): Tune thresholds for maximum sensitivity
```
python day3/threshold_tuner.py --model neoscreen_v1.tflite --data_dir ./dataset
```
Tries all threshold combinations. Prints the one that achieves >= 95% sensitivity.
Updates ml/inference.py automatically with the best values.

---

## Task 4 (1 hr): Cross-validate across Fitzpatrick skin types
```
python day3/fitzpatrick_test.py --model neoscreen_v1.tflite
```
Tests model on synthetic images across 6 skin tone types.
Confirms sclera analysis is skin-tone independent.

---

## Task 5 (2 hrs): Re-evaluate with final thresholds
```
python ml/evaluate.py --model neoscreen_v1.tflite --test_dir ./dataset
```
Target:
  Sensitivity >= 95%
  Kappa > 0.80
  AUC > 0.92

---

## End of Day 3 Checklist
- [ ] Real NIH images added to dataset
- [ ] Model retrained on combined data
- [ ] Sensitivity >= 95% confirmed
- [ ] Fitzpatrick test passes for all 6 types
- [ ] eval_results.json updated
- [ ] git commit -m "day3: real data + threshold tuning, sensitivity >= 95%"
