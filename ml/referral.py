import os
import sqlite3
from typing import Tuple

from geopy.distance import geodesic
from twilio.rest import Client


def get_nearest_phc(
    lat: float,
    lon: float,
    db_path: str = "data/phc_maharashtra.db",
) -> Tuple[str, str, float, float]:
    """
    Query SQLite for the nearest PHC using Haversine distance.

    Returns:
        (name, phone, phc_lat, phc_lon)
    """
    conn = sqlite3.connect(db_path)
    phcs = conn.execute("SELECT name, phone, lat, lon FROM phcs").fetchall()
    conn.close()

    if not phcs:
        raise RuntimeError("PHC database is empty. Check data/phc_maharashtra.db")

    nearest = min(phcs, key=lambda p: geodesic((lat, lon), (p[2], p[3])).km)
    distance_km = geodesic((lat, lon), (nearest[2], nearest[3])).km
    print(f"Nearest PHC: {nearest[0]}  ({distance_km:.1f} km away)")
    return nearest


def send_phc_alert(
    risk: str,
    baby_age_hrs: int,
    asha_id: str,
    lat: float,
    lon: float,
    db_path: str = "data/phc_maharashtra.db",
) -> bool:
    """
    Send automated SMS to nearest PHC. Only fires on HIGH risk.

    Returns True if message sent, False otherwise.
    Requires env vars: TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM
    """
    if risk != "HIGH":
        print(f"Risk is {risk} — no PHC alert sent.")
        return False

    phc = get_nearest_phc(lat, lon, db_path)
    phc_name, phc_phone, *_ = phc

    msg = (
        f"NeoScreen HIGH RISK ALERT | "
        f"Baby age: {baby_age_hrs}h | "
        f"ASHA: {asha_id} | "
        f"Location: {lat:.5f},{lon:.5f} | "
        f"PHC: {phc_name}"
    )

    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_TOKEN")
    from_number = os.environ.get("TWILIO_FROM")

    if not all([sid, token, from_number]):
        raise EnvironmentError(
            "Set TWILIO_SID, TWILIO_TOKEN, and TWILIO_FROM environment variables."
        )

    client = Client(sid, token)
    message = client.messages.create(
        body=msg,
        from_=from_number,
        to=phc_phone,
    )
    print(f"SMS sent to {phc_name} ({phc_phone}) | SID: {message.sid}")
    return True


if __name__ == "__main__":
    # Quick local test (requires .env or exported env vars)
    from dotenv import load_dotenv
    load_dotenv()

    send_phc_alert(
        risk="HIGH",
        baby_age_hrs=36,
        asha_id="MH-ASHA-001",
        lat=18.5204,
        lon=73.8567,
    )