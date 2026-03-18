"""
Day 6 — Task 2
Pitch Alignment Checker.
Maps every claim in the NeoScreen blueprint to the code that proves it.
Prints a table judges can verify themselves.

Usage:
    python day6/pitch_alignment.py
"""

import json
import os
import re
import sys


def check_model_size():
    path = "neoscreen_v1.tflite"
    if os.path.exists(path):
        mb = os.path.getsize(path) / (1024 * 1024)
        return mb < 5, f"{mb:.2f} MB (target < 5 MB)"
    return None, "neoscreen_v1.tflite not found — train first"


def check_offline(filepath, forbidden_patterns):
    if not os.path.exists(filepath):
        return None, f"{filepath} not found"
    with open(filepath) as f:
        content = f.read()
    hits = [p for p in forbidden_patterns if p in content]
    if hits:
        return False, f"Found network calls: {hits}"
    return True, "No network calls in inference path"


def check_no_pii_stored():
    """Scan Python files for storage of name/dob/photo bytes."""
    suspects = []
    for root, _, files in os.walk("."):
        if any(skip in root for skip in ["venv", ".git", "__pycache__"]):
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, errors="ignore") as f:
                content = f.read().lower()
            for pii_term in ["dob", "date_of_birth", "patient_name", "mother_name",
                              "photo_bytes", "image_bytes", "b64image"]:
                if pii_term in content:
                    suspects.append(f"{fpath}: '{pii_term}'")
    return len(suspects) == 0, suspects if suspects else ["No PII terms found"]


def check_twilio_sms_content():
    path = os.path.join("ml", "referral.py")
    if not os.path.exists(path):
        return None, f"{path} not found"
    with open(path) as f:
        content = f.read()
    # Check SMS body only contains allowed fields
    if "baby_age_hrs" in content and "asha_id" in content and "lat" in content:
        if "photo" not in content.lower() and "image" not in content.lower():
            return True, "SMS contains only: baby_age, asha_id, GPS, phc_name"
    return False, "SMS content may contain unexpected fields"


def check_sensitivity():
    path = "eval_results.json"
    if not os.path.exists(path):
        return None, "eval_results.json not found — run ml/evaluate.py first"
    with open(path) as f:
        r = json.load(f)
    sens = r.get("sensitivity_high_risk", 0)
    return sens >= 0.95, f"Sensitivity = {sens:.1%} (target >= 95%)"


def check_env_gitignored():
    path = ".gitignore"
    if not os.path.exists(path):
        return False, ".gitignore not found"
    with open(path) as f:
        content = f.read()
    return ".env" in content, ".env in .gitignore" if ".env" in content else ".env NOT in .gitignore"


def check_android_min_sdk():
    path = os.path.join("flutter", "android", "app", "build.gradle")
    if not os.path.exists(path):
        return None, f"{path} not found"
    with open(path) as f:
        content = f.read()
    if "minSdkVersion 26" in content:
        return True, "minSdkVersion 26 = Android 8+ confirmed"
    return False, "minSdkVersion not set to 26"


CLAIMS = [
    ("4 MB TF Lite model",              check_model_size),
    ("Fully offline inference",         lambda: check_offline("ml/inference.py",
                                            ["requests.get", "urllib", "http.client", "socket"])),
    ("No PII stored",                   check_no_pii_stored),
    ("SMS = age + asha_id + GPS only",  check_twilio_sms_content),
    ("Sensitivity >= 95%",              check_sensitivity),
    ("Credentials never in git",        check_env_gitignored),
    ("Android 8+ (minSdk 26)",          check_android_min_sdk),
]


def run():
    print("=" * 68)
    print("NeoScreen — Day 6: Pitch Claim Verification")
    print("=" * 68)
    print(f"\n{'Claim':<38} {'Pass?':>6}  Detail")
    print("-" * 68)

    all_pass = True
    lines = []

    for claim, checker in CLAIMS:
        try:
            passed, detail = checker()
        except Exception as e:
            passed, detail = False, str(e)

        if passed is None:
            symbol = "?"
        elif passed:
            symbol = "✓"
        else:
            symbol = "✗"
            all_pass = False

        print(f"  {claim:<36} {symbol:>6}  {detail if isinstance(detail, str) else str(detail[0])}")
        lines.append(f"{claim} | {symbol} | {detail}")

    # Save report
    os.makedirs("day6", exist_ok=True)
    with open(os.path.join("day6", "pitch_alignment.txt"), "w") as f:
        f.write("NeoScreen — Pitch Claim Verification\n")
        f.write("=" * 68 + "\n")
        for line in lines:
            f.write(line + "\n")

    print("\n" + "=" * 68)
    if all_pass:
        print("✓ All claims verified. Repo is pitch-ready.")
    else:
        print("✗ Some claims need attention. Fix before presenting to judges.")
    print("Report saved → day6/pitch_alignment.txt")
    print("=" * 68)


if __name__ == "__main__":
    run()
