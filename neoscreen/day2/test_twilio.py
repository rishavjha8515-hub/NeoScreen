"""
Day 2 — Task 10 (Afternoon, 1 hr)
Test the Twilio High Risk SMS alert end-to-end.
Sends a REAL test SMS — check your Twilio console after running.

Requires .env with:
    TWILIO_SID=ACxxxxxxxx
    TWILIO_TOKEN=xxxxxxxxx
    TWILIO_FROM=+1XXXXXXXXXX

Usage:
    python day2/test_twilio.py [--dry-run]
"""

import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv()


def check_env():
    missing = []
    for var in ["TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM"]:
        if not os.environ.get(var):
            missing.append(var)
    return missing


def run(dry_run: bool = False):
    print("=" * 56)
    print("NeoScreen — Day 2 Task 10: Twilio SMS Test")
    print("=" * 56)

    # ── Check credentials ─────────────────────────────────────────────────────
    print("\n[1/3] Checking .env credentials...")
    missing = check_env()
    if missing:
        print(f"  ✗  Missing env vars: {', '.join(missing)}")
        print("\n  Fix:")
        print("    1. Copy .env.example to .env")
        print("    2. Fill in your Twilio credentials from https://console.twilio.com")
        print("    3. Re-run this script")
        sys.exit(1)

    sid = os.environ["TWILIO_SID"]
    print(f"  ✓  TWILIO_SID:   {sid[:8]}...{sid[-4:]}")
    print(f"  ✓  TWILIO_TOKEN: ****")
    print(f"  ✓  TWILIO_FROM:  {os.environ['TWILIO_FROM']}")

    # ── Find nearest PHC to Pune ──────────────────────────────────────────────
    print("\n[2/3] Finding nearest PHC to test location (Pune)...")
    from ml.referral import get_nearest_phc

    db_path = os.path.join("data", "phc_maharashtra.db")
    if not os.path.exists(db_path):
        import subprocess
        subprocess.run([sys.executable, "scripts/seed_phc_db.py"], check=True)

    phc = get_nearest_phc(18.5204, 73.8567, db_path)
    print(f"  ✓  Nearest PHC: {phc[0]}")
    print(f"     Phone:       {phc[1]}")

    # ── Send test SMS ─────────────────────────────────────────────────────────
    print(f"\n[3/3] {'[DRY RUN] ' if dry_run else ''}Sending test SMS...")

    test_msg = (
        "NeoScreen TEST ALERT | Baby age: 36h | "
        "ASHA: TEST-ASHA-001 | Location: 18.52,73.85 | "
        f"PHC: {phc[0]} | THIS IS A TEST"
    )
    print(f"  Message: {test_msg}")

    if dry_run:
        print("\n  [DRY RUN] SMS not actually sent.")
        print("  Remove --dry-run to send a real SMS.")
    else:
        try:
            from twilio.rest import Client
            client = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
            message = client.messages.create(
                body=test_msg,
                from_=os.environ["TWILIO_FROM"],
                to=phc[1],
            )
            print(f"\n  ✓  SMS sent! SID: {message.sid}")
            print(f"     Status: {message.status}")
            print(f"     Check: https://console.twilio.com/us1/monitor/logs/sms")
        except Exception as e:
            print(f"\n  ✗  SMS failed: {e}")
            print("\n  Common fixes:")
            print("    - Check TWILIO_SID and TWILIO_TOKEN in .env")
            print("    - Make sure your Twilio number is SMS-capable")
            print("    - Free trial accounts can only SMS verified numbers")
            sys.exit(1)

    print("\n" + "=" * 56)
    print("Twilio test complete.")
    print("Next: python day2/test_end_to_end.py --model neoscreen_v1.tflite")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without sending a real SMS")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
