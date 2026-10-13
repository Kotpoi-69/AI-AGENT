import os
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# FIREBASE
# =========================================================

FIREBASE_URL = os.getenv("FIREBASE_URL")


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# =========================================================
# OPENWEATHER
# =========================================================

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

LAT = float(
    os.getenv("LAT", "-6.484519")
)

LON = float(
    os.getenv("LON", "106.810065")
)


# =========================================================
# HARDWARE SAFETY
# =========================================================

# Batas maksimum durasi pompa.
# Bukan aturan budidaya.
MAX_PUMP_DURATION = 20