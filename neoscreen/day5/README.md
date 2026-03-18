# Day 5 — Wednesday: Real Device Testing + Accuracy Report

Goal: APK tested on real Android device. Accuracy report written. Demo video recorded.

---

## Task 1 (2 hrs): Install APK on real device

```
# Connect Android phone via USB
# Enable Developer Options → USB Debugging on phone first

cd flutter
flutter devices           # confirm device is listed
flutter install           # installs release APK
```

If no device available, use Android emulator:
```
flutter emulators --launch Pixel_6_API_33
flutter run
```

---

## Task 2 (1 hr): Real device benchmark
```
# Run on connected device
flutter run --profile
# Press 'P' in terminal to enable performance overlay
# Record frame times in day5/real_device_results.txt
```

---

## Task 3 (2 hrs): Clinical accuracy comparison
```
python day5/accuracy_report.py --model neoscreen_v1.tflite --data_dir ./dataset
```
Generates day5/NeoScreen_Accuracy_Report.txt with:
- Full confusion matrix
- Sensitivity / Specificity / Kappa / AUC
- Comparison to Bilicam paper benchmarks (97% sensitivity)
- Honest gap analysis

---

## Task 4 (2 hrs): Record demo video
Screen record on device showing:
1. Open app → camera screen
2. Point at baby eye photo (print a test image)
3. Wait < 30 seconds
4. See High Risk result in Hindi
5. SMS sent to PHC (show Twilio console)

Save as: day5/demo_video.mp4

---

## Task 5 (1 hr): Write honest limitations section for README
```
python day5/update_readme.py
```
Adds to README.md:
- Current model trained on synthetic data only
- Clinical validation pending (IISc Medical Network)
- Not a replacement for blood test
- Sensitivity target met / not met (auto-filled from eval_results.json)

---

## End of Day 5 Checklist
- [ ] APK runs on real device or emulator
- [ ] Inference < 500ms confirmed on device
- [ ] day5/NeoScreen_Accuracy_Report.txt written
- [ ] Demo video recorded
- [ ] README updated with real accuracy numbers
- [ ] git commit -m "day5: real device test, accuracy report, demo video"
