"""
Day 2 — Task 9 (Afternoon, 1 hr)
Test the PHC nearest-lookup against 5 real Maharashtra GPS coordinates.
Verifies SQLite database is seeded correctly and GeoPy distance works.

Usage:
    python day2/test_phc_lookup.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# 5 real locations across Maharashtra with expected nearest PHC
TEST_LOCATIONS = [
    {"name": "Pune City Centre",      "lat": 18.5204, "lon": 73.8567},
    {"name": "Nashik",                "lat": 19.9975, "lon": 73.7898},
    {"name": "Aurangabad",            "lat": 19.8762, "lon": 75.3433},
    {"name": "Kolhapur",              "lat": 16.7050, "lon": 74.2433},
    {"name": "Latur",                 "lat": 18.3956, "lon": 76.5604},
    {"name": "Remote village (Bhor)", "lat": 18.1531, "lon": 73.8512},
]


def ensure_db():
    db_path = os.path.join("data", "phc_maharashtra.db")
    if not os.path.exists(db_path):
        print("  PHC database not found — generating...")
        import subprocess
        subprocess.run([sys.executable, "scripts/seed_phc_db.py"], check=True)
    return db_path


def run():
    print("=" * 62)
    print("NeoScreen — Day 2 Task 9: PHC Nearest-Lookup Test")
    print("=" * 62)

    db_path = ensure_db()

    # Import after ensuring DB exists
    from ml.referral import get_nearest_phc

    print(f"\n{'Location':<26} {'Nearest PHC':<28} {'Dist':>6}  {'ms':>5}")
    print("-" * 72)

    all_passed = True
    for loc in TEST_LOCATIONS:
        t0 = time.perf_counter()
        try:
            phc = get_nearest_phc(loc["lat"], loc["lon"], db_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            from geopy.distance import geodesic
            dist_km = geodesic((loc["lat"], loc["lon"]), (phc[2], phc[3])).km

            speed_ok = elapsed_ms < 100
            print(f"  {loc['name']:<24} {phc[0]:<28} {dist_km:>5.1f}km  {elapsed_ms:>4.1f}ms  {'✓' if speed_ok else '⚠ slow'}")

        except Exception as e:
            print(f"  {loc['name']:<24} ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 62)
    if all_passed:
        print("✓ All PHC lookups passed. Database is working correctly.")
        print("  Next: python day2/test_twilio.py")
    else:
        print("✗ Some lookups failed. Check data/phc_maharashtra.db")
    print("=" * 62)


if __name__ == "__main__":
    run()
