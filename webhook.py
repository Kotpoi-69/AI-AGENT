# webhook.py

from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from datetime import datetime
from zoneinfo import ZoneInfo
from firebase_service import get_data, update_data
from whatsapp_service import send_whatsapp, send_whatsapp_chunked

from intent_classifier import classify_intent
from agri_ai_agent import agriculture_chat_agent
from gemini_service import analyze_farm
import re
app = FastAPI()


# =========================
# WHATSAPP SENDER SAFE
# =========================

def send_smart(to, msg):
    try:
        if not isinstance(msg, str):
            msg = str(msg)

        if len(msg) > 1500:
            result = send_whatsapp_chunked(to, msg)
        else:
            result = send_whatsapp(to, msg)

        if result is False:
            print("WHATSAPP WARNING: Pesan gagal dikirim.")

            return False

        return True

    except Exception as e:
        print(
            f"WHATSAPP SEND ERROR: "
            f"{type(e).__name__}: {e}"
        )
        return False


def parse_set_tanaman(message):

    match = re.match(
        r"set tanaman\s+(\d+)",
        message.lower()
    )

    if match:
        return int(match.group(1))

    return None

def parse_set_umur(message):

    match = re.match(
        r"set umur\s+(\d+)",
        message.lower()
    )

    if match:
        return int(match.group(1))

    return None



# =========================
# HEALTH CHECK
# =========================
@app.get("/webhook")
async def health():
    return {"status": "ok"}


# =========================
# MAIN WEBHOOK
# =========================
@app.post("/webhook")
async def webhook(From: str = Form(...), Body: str = Form(...)):

    sender = From.replace("whatsapp:", "")
    message = Body.strip()
    plant_count = parse_set_tanaman(message)
    age_days = parse_set_umur(message)

    if plant_count is not None:

        update_data(
            "agent",
            {
                "plant_count": plant_count
            }
        )

        send_smart(
            sender,
            f"✅ Jumlah tanaman berhasil diset menjadi {plant_count} tanaman."
        )

        return PlainTextResponse("OK")
    
    if age_days is not None:

        today = datetime.now(
            ZoneInfo("Asia/Jakarta")
        ).date().isoformat()

        success = update_data(
        "agent",
        {
            "age_days": age_days,
            "age_reference_days": age_days,
            "age_reference_date": today
        }
    )
    if success:
        send_smart(
            sender,
            (
                f"✅ Umur tanaman berhasil diset menjadi "
                f"{age_days} hari.\n"
                f"📅 Tanggal acuan: {today}\n"
                "Umur akan bertambah otomatis setiap pergantian hari."
            )
        )
    else:
        send_smart(
            sender,
            "❌ Gagal menyimpan umur tanaman ke Firebase."
        )

    return PlainTextResponse("OK")

    print("=" * 60)
    print("FROM:", sender)
    print("MSG :", message)
    print("=" * 60)

    try:
        # =========================
        # LOAD FIREBASE DATA
        # =========================
        sensor = get_data("sensor") or {}
        weather = get_data("weather") or {}
        agent = get_data("agent") or {}
        cam = get_data("cam") or {}


        # =========================
        # INTENT CLASSIFICATION
        # =========================
        intent = classify_intent(message)
        print("INTENT:", intent)

        # =====================================================
        # 1. COMMAND MODE
        # =====================================================
        if intent["intent"] == "command":

            cmd = intent["command"]

            if cmd == "status":
                reply = f"""
🌶️ STATUS LAHAN
Jumlah Tanaman: {agent.get('plant_count',0)}
Umur: {agent.get('age_days',0)} hari
Soil: {sensor.get('soil',0)} %
Suhu: {sensor.get('temperature',0)} °C
Humidity: {sensor.get('humidity',0)} %
pH: {sensor.get('ph',0)}

Status AI: {agent.get('status','-')}
Pompa: {agent.get('pump_duration',0)} detik
"""

            elif cmd == "cuaca":
                reply = f"""
☁️ CUACA

Forecast: {weather.get('forecast','-')}
Suhu: {weather.get('temperature',0)} °C
Humidity: {weather.get('humidity',0)} %
Rain Prob: {weather.get('rain_probability',0)} %
"""

            elif cmd == "pompa":
                reply = f"""
🚿 POMPA

Status: {agent.get('status','-')}
Durasi: {agent.get('pump_duration',0)} detik
Alasan: {agent.get('reason','-')}
"""

            elif cmd == "tanaman":
                reply = f"""
🌿 TANAMAN

Terdeteksi: {'Ya' if cam.get('tanaman') else 'Tidak'}
Confidence: {cam.get('confidence',0)}
"""

 
            else:
                reply = "Command tidak dikenali."

            send_smart(sender, reply)
            return PlainTextResponse("OK")

        # =====================================================
        # 2. ANALYSIS MODE (AI INSIGHT FIXED)
        # =====================================================
        elif intent["intent"] == "analysis":

            try:
                ai_input = {
                    "soil": sensor.get("soil", 0),
                    "temperature": sensor.get("temperature", 0),
                    "humidity": sensor.get("humidity", 0),
                    "ph": sensor.get("ph", 0),
                    "rain_probability": weather.get("rain_probability", 0),
                    "status": agent.get("status", "-"),
                    "pump_duration": agent.get("pump_duration", 0)
                }

                reply = analyze_farm(ai_input)

            except Exception as e:
                print("ANALYSIS ERROR:", e)
                reply = "AI analisis sedang tidak tersedia, coba lagi nanti."

            send_smart(sender, reply)
            return PlainTextResponse("OK")

       
            
        # =====================================================
        # 3. CHAT MODE (FREE AGENT)
        # =====================================================
        else:

            try:
                reply = agriculture_chat_agent(message)

            except Exception as e:
                print("CHAT ERROR:", e)
                reply = "AI chat sedang sibuk, coba lagi nanti."

            send_smart(sender, reply)
            return PlainTextResponse("OK")

    
    except Exception as e:
        print(
            f"WEBHOOK FATAL: "
            f"{type(e).__name__}: {e}"
        )

        # Jangan mencoba mengirim pesan error melalui
        # Twilio lagi karena pengirimannya mungkin sedang gagal.
        return PlainTextResponse("Internal error", status_code=500)
