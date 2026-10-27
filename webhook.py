# webhook.py (BAGIAN 1)

import re
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form, Response
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from firebase_service import get_data, update_data
from whatsapp_service import send_whatsapp, send_whatsapp_chunked

from intent_classifier import classify_intent
from agri_ai_agent import agriculture_chat_agent
from gemini_service import analyze_farm

app = FastAPI(
    title="Smart Agriculture WhatsApp Webhook",
)

def send_smart_twiml(msg):
    twilio_response = MessagingResponse()
    if not isinstance(msg, str):
        msg = str(msg)
    twilio_response.message(msg)
    return Response(content=str(twilio_response), media_type="application/xml")

def parse_set_tanaman(message):
    match = re.fullmatch(r"set tanaman\s+(\d+)", message.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None

def parse_set_umur(message):
    match = re.fullmatch(r"set umur\s+(\d+)", message.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None

def as_dict(value):
    return value if isinstance(value, dict) else {}

def to_epoch_seconds(value):
    if value is None: return None
    try:
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 100_000_000_000: timestamp /= 1000
            return timestamp
        if isinstance(value, str):
            value = value.strip()
            if not value: return None
            try:
                timestamp = float(value)
                if timestamp > 100_000_000_000: timestamp /= 1000
                return timestamp
            except ValueError: pass
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    except: return None
    return None

def get_latest_fertilization_record(history):
    if isinstance(history, list): records = [item for item in history if isinstance(item, dict)]
    elif isinstance(history, dict):
        if isinstance(history.get("latest"), dict): return history["latest"]
        if any(k in history for k in ("timestamp", "date", "waktu")): return history
        records = [item for item in history.values() if isinstance(item, dict)]
    else: return {}
    return max(records, key=lambda r: to_epoch_seconds(r.get("timestamp")) or 0) if records else {}

def calculate_days_since_fertilization(record):
    if not isinstance(record, dict): return None
    timestamp = to_epoch_seconds(record.get("timestamp"))
    if timestamp is None: return None
    elapsed = datetime.now(timezone.utc).timestamp() - timestamp
    return elapsed / 86400 if elapsed >= 0 else None

def answer_fertilization_date_question(message):
    if not (re.search(r"\b(kapan|tanggal|waktu)\b", message, flags=re.IGNORECASE) and re.search(r"\b(pemupukan|pupuk)\b", message, flags=re.IGNORECASE)): return None
    history = as_dict(get_data("fertilization_history/latest"))
    if not history: return "Riwayat pemupukan terakhir belum ditemukan di Firebase."
    return f"🌱 Pemupukan terakhir tercatat pada {history.get('last_applied_date', '-')}"

def build_gemini_context(sensor, weather, agent, cam):
    plant_detection = as_dict(get_data("plant_detection")) or {"plant_detected": cam.get("tanaman", False), "confidence": cam.get("confidence", 0)}
    latest_fertilization = get_latest_fertilization_record(get_data("fertilization_history"))
    return {
        "sensor": {"soil": sensor.get("soil", 0), "temperature": sensor.get("temperature", 0), "humidity": sensor.get("humidity", 0), "ph": sensor.get("ph", 0), "rain": sensor.get("rain", False)},
        "weather": {"forecast": weather.get("forecast", "-"), "temperature": weather.get("temperature", 0), "humidity": weather.get("humidity", 0), "rain_probability": weather.get("rain_probability", 0), "rain_3h": weather.get("rain_3h", 0)},
        "plant": {"plant_count": agent.get("plant_count", 0), "age_days": agent.get("age_days", 0)},
        "plant_detection": plant_detection,
        "fertilization": {"days_since_last": calculate_days_since_fertilization(latest_fertilization)},
        "agent": {"status": agent.get("status", "-"), "pump_duration": agent.get("pump_duration", 0)},
    }

def format_analysis_result(result):
    if not isinstance(result, dict): return str(result)
    action_labels = {"NONE": "Belum ada tindakan", "IRRIGATE": "Rekomendasi irigasi", "FERTILIZE": "Rekomendasi pemupukan"}
    return f"🌶️ ANALISIS AI LAHAN\n\nRekomendasi: {action_labels.get(str(result.get('action')).upper(), 'NONE')}\nAlasan: {result.get('reason', '-')}"

# webhook.py (BAGIAN 2)

@app.get("/")
async def root():
    return {"status": "ok", "service": "smart-agriculture-webhook"}

@app.get("/webhook")
async def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(
    From: str = Form(..., alias="From"),
    Body: str = Form(..., alias="Body")
):
    sender = From.replace("whatsapp:", "")
    message = Body.strip()

    if not message:
        return send_smart_twiml("Pesan kosong. Silakan kirim perintah.")

    print("=" * 60)
    print("FROM:", sender)
    print("MSG :", message)
    print("=" * 60)

    plant_count = parse_set_tanaman(message)
    if plant_count is not None:
        try:
            success = update_data("agent", {"plant_count": plant_count})
            reply = f"✅ Jumlah tanaman berhasil diset menjadi {plant_count}." if success else "❌ Gagal menyimpan ke Firebase."
        except Exception as e:
            reply = "❌ Terjadi kesalahan sistem."
        return send_smart_twiml(reply)

    age_days = parse_set_umur(message)
    if age_days is not None:
        today = datetime.now(ZoneInfo("Asia/Jakarta")).date().isoformat()
        try:
            success = update_data("agent", {"age_days": age_days, "age_reference_days": age_days, "age_reference_date": today})
            reply = f"✅ Umur tanaman berhasil diset menjadi {age_days} hari." if success else "❌ Gagal menyimpan ke Firebase."
        except Exception as e:
            reply = "❌ Terjadi kesalahan sistem."
        return send_smart_twiml(reply)

    try:
        sensor = as_dict(get_data("sensor"))
        weather = as_dict(get_data("weather"))
        agent = as_dict(get_data("agent"))
        cam = as_dict(get_data("cam"))

        fertilization_reply = answer_fertilization_date_question(message)
        if fertilization_reply is not None:
            return send_smart_twiml(fertilization_reply)

        intent = classify_intent(message)
        print("INTENT:", intent)

        if not isinstance(intent, dict):
            return send_smart_twiml("Maaf, perintah belum dapat dikenali.")

        intent_type = intent.get("intent", "chat")

        if intent_type == "command":
            cmd = intent.get("command")
            if cmd == "status":
                reply = (
                    "🌶️ STATUS LAHAN\n"
                    f"Jumlah Tanaman: {agent.get('plant_count', 0)}\n"
                    f"Umur: {agent.get('age_days', 0)} hari\n"
                    f"Soil: {sensor.get('soil', 0)} %\n"
                    f"Suhu: {sensor.get('temperature', 0)} °C\n"
                    f"Humidity: {sensor.get('humidity', 0)} %\n"
                    f"pH: {sensor.get('ph', 0)}\n\n"
                    f"Status AI: {agent.get('status', '-')}\n"
                    f"Pompa: {agent.get('pump_duration', 0)} detik"
                )
            elif cmd == "cuaca":
                reply = f"☁️ CUACA\n\nForecast: {weather.get('forecast', '-')}\nSuhu: {weather.get('temperature', 0)} °C"
            elif cmd == "pompa":
                reply = f"🚿 POMPA\n\nStatus: {agent.get('status', '-')}\nDurasi: {agent.get('pump_duration', 0)} detik"
            elif cmd == "tanaman":
                detected = cam.get("tanaman", cam.get("plant_detected", False))
                reply = f"🌿 DETEKSI TANAMAN\n\nTerdeteksi: {'Ya' if detected else 'Tidak'}"
            else:
                reply = "Perintah tidak dikenali."
            return send_smart_twiml(reply)

        elif intent_type == "analysis":
            try:
                context = build_gemini_context(sensor, weather, agent, cam)
                result = analyze_farm(context, purpose="analysis")
                reply = format_analysis_result(result)
            except Exception as e:
                reply = "AI analisis sedang tidak tersedia."
            
            if len(reply) > 1500:
                send_whatsapp_chunked(sender, reply)
                return Response(content=str(MessagingResponse()), media_type="application/xml")
            else:
                return send_smart_twiml(reply)

        else:
            try:
                reply = agriculture_chat_agent(message)
            except Exception as e:
                reply = "AI chat sedang sibuk."
            
            # 🟢 PROTEKSI BARU: Pecah chat jika respon Gemini > 1500 karakter
            if len(reply) > 1500:
                send_whatsapp_chunked(sender, reply)
                return Response(content=str(MessagingResponse()), media_type="application/xml")
            else:
                return send_smart_twiml(reply)

    except Exception as e:
        print(f"WEBHOOK FATAL: {type(e).__name__}: {e}")
        return send_smart_twiml("Terjadi kesalahan pada sistem. Silakan coba lagi nanti.")
