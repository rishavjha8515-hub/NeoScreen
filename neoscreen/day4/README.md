# Day 4 — Tuesday: Flutter UI Polish + Offline Hardening

Goal: App looks demo-ready. Works 100% offline. Handles every edge case.

---

## Task 1 (2 hrs): UI Polish
- Add NeoScreen logo/splash screen
- Add baby age input (hours since birth)
- Add ASHA worker ID input
- Improve result screen — bigger risk badge, animated confidence bars

```
# Edit flutter/lib/screens/main_screen.dart
# Edit flutter/lib/screens/result_screen.dart
# Then rebuild:
cd flutter && flutter build apk --debug
```

---

## Task 2 (1 hr): Offline hardening — no crash without internet
```
python day4/test_offline_mode.py
```
Disconnects network simulation and verifies:
- Inference still works
- PHC lookup still works (SQLite is local)
- Only SMS fails gracefully (shows "SMS queued for next connectivity")

---

## Task 3 (1 hr): Camera edge cases
```
python day4/test_camera_edge_cases.py --model neoscreen_v1.tflite
```
Tests:
- Very dark image → CLAHE enhancement → retry prompt
- Blurry image → retry prompt
- No eye detected → retry prompt
- Partial eye → still works

---

## Task 4 (2 hrs): Performance profiling on low-end device
```
python day4/benchmark.py --model neoscreen_v1.tflite --runs 50
```
Runs 50 inferences and reports:
- Mean / P95 / P99 latency
- Memory usage
- Confirms < 500ms on simulated Moto E13 constraints

---

## Task 5 (2 hrs): Add history screen to Flutter app
Stores last 10 screenings locally (SQLite).
No PII — only timestamp, risk level, baby age in hours.

---

## End of Day 4 Checklist
- [ ] App has ASHA ID + baby age input
- [ ] Offline mode tested — no crash
- [ ] Camera edge cases handled
- [ ] Benchmark passes (< 500ms P95)
- [ ] History screen shows last 10 screenings
- [ ] git commit -m "day4: UI polish, offline hardening, history screen"
