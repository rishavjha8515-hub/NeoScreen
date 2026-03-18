# Day 6 — Thursday: Pitch Prep + Documentation

Goal: GitHub repo is judge-ready. README tells the full story. Pitch deck aligned with code.

---

## Task 1 (2 hrs): Polish README for judges
```
python day6/polish_readme.py
```
Updates README.md with:
- Real accuracy numbers from eval_results.json
- Demo GIF placeholder link
- IISc Lab2Launch badge
- Architecture diagram (ASCII)
- Clean quickstart that actually works

---

## Task 2 (2 hrs): Generate pitch alignment doc
```
python day6/pitch_alignment.py
```
Creates day6/pitch_alignment.txt showing:
- Every claim in the blueprint → which code file proves it
- e.g. "4MB model" → neoscreen_v1.tflite file size
- e.g. "< 30 seconds" → benchmark results
- e.g. "offline" → no network calls in inference.py
- e.g. "Fitzpatrick independent" → day3/fitzpatrick_test.py results

---

## Task 3 (1 hr): Security audit — confirm no PII stored
```
python day6/security_audit.py
```
Scans all Python + Dart files and confirms:
- No photo bytes are transmitted
- No name/DOB stored anywhere
- SMS contains only: baby_age_hours, asha_id, GPS, risk_score
- .env is in .gitignore

---

## Task 4 (1 hr): Final GitHub cleanup
```
python day6/github_cleanup.py
```
- Verifies .gitignore covers all sensitive files
- Checks no credentials are hardcoded
- Adds GitHub Topics to repo description template

---

## Task 5 (2 hrs): Create a DEMO.md
```
python day6/create_demo_md.py
```
Creates DEMO.md with:
- Step-by-step judge walkthrough
- Screenshots placeholder (add real ones from Day 5)
- Expected outputs at each step
- Twilio console screenshot guide

---

## End of Day 6 Checklist
- [ ] README has real accuracy numbers
- [ ] pitch_alignment.txt links every claim to code
- [ ] security_audit.py passes — no PII leaks
- [ ] No hardcoded credentials anywhere
- [ ] DEMO.md created
- [ ] git commit -m "day6: judge-ready docs, pitch alignment, security audit"
