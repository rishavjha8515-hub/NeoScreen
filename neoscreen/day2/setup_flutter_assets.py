"""
Day 2 — Task 8
Copy trained model and PHC database into Flutter assets folder.
Run this from the project root after Day 1 is complete.

Usage:
    python day2/setup_flutter_assets.py
"""

import os
import shutil
import sys


def setup_assets():
    print("=" * 54)
    print("NeoScreen — Day 2: Flutter Asset Setup")
    print("=" * 54)

    # ── Paths ─────────────────────────────────────────────────────────────────
    MODEL_SRC = "neoscreen_v1.tflite"
    MODEL_DST = os.path.join("flutter", "assets", "models", "neoscreen_v1.tflite")

    DB_SRC = os.path.join("data", "phc_maharashtra.db")
    DB_DST = os.path.join("flutter", "assets", "data", "phc_maharashtra.db")

    ENV_SRC = ".env"
    ENV_DST = os.path.join("flutter", ".env")

    # ── Create asset dirs ─────────────────────────────────────────────────────
    os.makedirs(os.path.join("flutter", "assets", "models"), exist_ok=True)
    os.makedirs(os.path.join("flutter", "assets", "data"), exist_ok=True)

    errors = []

    # ── Copy model ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Copying TF Lite model...")
    if os.path.exists(MODEL_SRC):
        size_mb = os.path.getsize(MODEL_SRC) / (1024 * 1024)
        shutil.copy2(MODEL_SRC, MODEL_DST)
        print(f"  ✓  {MODEL_SRC}  ({size_mb:.2f} MB)  →  {MODEL_DST}")
    else:
        print(f"  ✗  {MODEL_SRC} not found.")
        print("     Run Day 1 training first: python ml/train.py")
        errors.append("model")

    # ── Copy PHC database ─────────────────────────────────────────────────────
    print(f"\n[2/3] Copying PHC database...")
    if not os.path.exists(DB_SRC):
        print(f"  Database not found at {DB_SRC}. Generating now...")
        import subprocess
        subprocess.run([sys.executable, "scripts/seed_phc_db.py"], check=True)

    if os.path.exists(DB_SRC):
        size_kb = os.path.getsize(DB_SRC) / 1024
        shutil.copy2(DB_SRC, DB_DST)
        print(f"  ✓  {DB_SRC}  ({size_kb:.1f} KB)  →  {DB_DST}")
    else:
        print(f"  ✗  Could not generate {DB_SRC}")
        errors.append("database")

    # ── Copy .env ─────────────────────────────────────────────────────────────
    print(f"\n[3/3] Copying .env for Flutter...")
    if os.path.exists(ENV_SRC):
        shutil.copy2(ENV_SRC, ENV_DST)
        print(f"  ✓  .env copied to flutter/.env")
    else:
        print(f"  ⚠   .env not found. Creating from template...")
        shutil.copy2(".env.example", ENV_DST)
        print(f"  ✓  flutter/.env created from .env.example")
        print(f"     Edit flutter/.env and fill in your Twilio credentials.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    if errors:
        print(f"Setup incomplete — missing: {', '.join(errors)}")
        print("Fix the above and re-run this script.")
        sys.exit(1)
    else:
        print("All assets ready. Next:")
        print("  cd flutter")
        print("  flutter pub get")
        print("  flutter test")
    print("=" * 54)


if __name__ == "__main__":
    setup_assets()
