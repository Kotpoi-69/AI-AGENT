
# webhook.py

import re
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from firebase_service import get_data, update_data

from intent_classifier import classify_intent
from agri_ai_agent import agriculture_chat_agent
from gemini_service import analyze_farm
from xml.etree import ElementTree as ET




app = FastAPI(
    title="Smart Agriculture WhatsApp Webhook",
)

# ============================================================
# FASTAPI
# ============================================================



# ============================================================
# WHATSAPP
# ============================================================

def send_smart_twiml(message=None):
    twiml = MessagingResponse()

    if message:
        # Satu pesan untuk respons yang panjangnya masih sesuai batas.
        twiml.message(str(message))

    xml_response = str(twiml)

    # Validasi XML sebelum dikirim.
    try:
        root = ET.fromstring(xml_response)
        print("TWIML XML VALID:", root.tag == "Response")
        print("TWIML MESSAGE COUNT:", len(root.findall("Message")))
        print("TWIML BODY LENGTH:", sum(
            len(node.text or "")
            for node in root.findall(".//Body")
        ))
    except ET.ParseError as exc:
        print("TWIML XML INVALID:", repr(exc))
        raise

    print("TWIML FULL LENGTH:", len(xml_response))
    print("TWIML END:", repr(xml_response[-300:]))

    return Response(
        content=xml_response,
        media_type="application/xml",
        status_code=200
    )


# ============================================================
# PARSE COMMAND
# ============================================================

def parse_set_tanaman(message):
    match = re.fullmatch(
        r"set tanaman\s+(\d+)",
        message.strip(),
        flags=re.IGNORECASE
    )

    return int(match.group(1)) if match else None


def parse_set_umur(message):
    match = re.fullmatch(
        r"set umur\s+(\d+)",
        message.strip(),
        flags=re.IGNORECASE
    )

    return int(match.group(1)) if match else None


# ============================================================
# DATA HELPERS
# ============================================================

def as_dict(value):
    """Pastikan data Firebase berbentuk dictionary."""

    return value if isinstance(value, dict) else {}


def to_epoch_seconds(value):
    """
    Konversi timestamp menjadi epoch seconds.

    Mendukung:
    - Unix timestamp dalam detik
    - Unix timestamp dalam milidetik
    - ISO datetime, misalnya 2026-10-13T10:00:00Z

    Jika format tidak dikenal, kembalikan None.
    """

    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            timestamp = float(value)

            # Timestamp milidetik
            if timestamp > 100_000_000_000:
                timestamp /= 1000

            return timestamp

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

            # Timestamp numerik berbentuk string
            try:
                timestamp = float(value)

                if timestamp > 100_000_000_000:
                    timestamp /= 1000

                return timestamp
            except ValueError:
                pass

            # ISO datetime
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            return parsed.timestamp()

    except (ValueError, TypeError, OverflowError):
        return None

    return None


def get_latest_fertilization_record(history):
    """
    Ambil catatan pemupukan terbaru dari struktur Firebase
    yang umum digunakan.

    Struktur yang didukung antara lain:
    - {"latest": {...}}
    - {"timestamp": ..., ...}
    - {"record_1": {...}, "record_2": {...}}
    - daftar record
    """

    if isinstance(history, list):
        records = [item for item in history if isinstance(item, dict)]

    elif isinstance(history, dict):
        if isinstance(history.get("latest"), dict):
            return history["latest"]

        # Data sudah berupa satu catatan
        date_keys = (
            "timestamp",
            "date",
            "datetime",
            "created_at",
            "waktu",
            "tanggal",
            "last_applied_at",
            "last_applied_date"
        )

        if any(key in history for key in date_keys):
            return history

        records = [
            item for item in history.values()
            if isinstance(item, dict)
        ]

    else:
        return {}

    if not records:
        return {}

    def record_timestamp(record):
        for key in (
            "timestamp",
            "date",
            "datetime",
            "created_at",
            "waktu",
            "tanggal",
            "last_applied_at",
            "last_applied_date"
        ):
            timestamp = to_epoch_seconds(record.get(key))

            if timestamp is not None:
                return timestamp

        return 0

    return max(records, key=record_timestamp)


def calculate_days_since_fertilization(record):
    """Hitung jarak hari sejak catatan pemupukan terakhir."""

    if not isinstance(record, dict):
        return None

    # Gunakan nilai yang sudah dihitung jika tersedia.
    for key in ("days_since_last", "days_since_last_fertilization"):
        value = record.get(key)

        if value is not None:
            try:
                days = float(value)

                if days >= 0:
                    return days
            except (ValueError, TypeError):
                pass

    timestamp = None

    for key in (
        "timestamp",
        "date",
        "datetime",
        "created_at",
        "waktu",
        "tanggal",
        "last_applied_at",
        "last_applied_date"
    ):
        timestamp = to_epoch_seconds(record.get(key))

        if timestamp is not None:
            break

    if timestamp is None:
        return None

    elapsed_seconds = datetime.now(timezone.utc).timestamp() - timestamp

    if elapsed_seconds < 0:
        return None

    return elapsed_seconds / 86400


def answer_fertilization_date_question(message):
    """
    Jawab pertanyaan tanggal pemupukan langsung dari Firebase.
    Tidak bergantung pada Gemini atau intent classifier.
    """

    asks_when = re.search(
        r"\b(kapan|tanggal|jam|pukul|waktu)\b",
        message,
        flags=re.IGNORECASE
    )

    mentions_fertilization = re.search(
        r"\b(pemupukan|pupuk|memupuk|dipupuk)\b",
        message,
        flags=re.IGNORECASE
    )

    if not (asks_when and mentions_fertilization):
        return None

    history = as_dict(
        get_data("fertilization_history/latest")
    )

    if not history:
        return (
            "Riwayat pemupukan terakhir belum ditemukan "
            "di Firebase pada /fertilization_history/latest."
        )

    applied_date = history.get("last_applied_date")
    timestamp = to_epoch_seconds(
        history.get("last_applied_at", history.get("timestamp"))
    )

    # Utamakan tanggal yang tersimpan di Firebase.
    readable_date = None

    if applied_date:
        try:
            parsed_date = date.fromisoformat(str(applied_date))
            readable_date = parsed_date.strftime("%d-%m-%Y")
        except ValueError:
            pass

    # Jika timestamp tersedia, tampilkan jam juga.
    if timestamp is not None:
        try:
            applied_datetime = datetime.fromtimestamp(
                timestamp,
                tz=ZoneInfo("Asia/Jakarta")
            )

            readable_datetime = applied_datetime.strftime(
                "%d-%m-%Y pukul %H:%M WIB"
            )

            return (
                f"🌱 Pemupukan terakhir tercatat pada "
                f"{readable_datetime}.\n"
                "Sumber: Firebase "
                "/fertilization_history/latest."
            )
        except (ValueError, OverflowError, OSError):
            pass

    if readable_date:
        return (
            f"🌱 Pemupukan terakhir tercatat pada "
            f"{readable_date}.\n"
            "Jam pemupukan tidak tersedia atau tidak valid.\n"
            "Sumber: Firebase "
            "/fertilization_history/latest."
        )

    return (
        "Catatan pemupukan ditemukan, tetapi tanggal "
        "dan timestamp pemupukan tidak valid."
    )

def build_gemini_context(sensor, weather, agent, cam):
    """
    Bentuk konteks bertingkat yang diharapkan gemini_service.py.

    Catatan:
    Path plant_detection dan fertilization_history di bawah
    harus sesuai dengan struktur Firebase yang sebenarnya.
    """

    # Data deteksi tanaman, jika disimpan terpisah.
    plant_detection = as_dict(
        get_data("plant_detection")
    )

    # Fallback ke data kamera lama jika path terpisah kosong.
    if not plant_detection:
        plant_detection = {
            "plant_detected": cam.get(
                "plant_detected",
                cam.get("tanaman", False)
            ),
            "confidence": cam.get("confidence", 0),
        }

        # Jangan membuat timestamp palsu.
        for key in ("timestamp", "updated_at", "created_at"):
            if cam.get(key) is not None:
                plant_detection["timestamp"] = cam[key]
                break

    # Normalisasi nama field deteksi.
    if "plant_detected" not in plant_detection:
        plant_detection["plant_detected"] = plant_detection.get(
            "tanaman", False
        )

    detection_timestamp = to_epoch_seconds(
        plant_detection.get("timestamp")
    )

    if detection_timestamp is None:
        for key in ("updated_at", "created_at"):
            detection_timestamp = to_epoch_seconds(
                plant_detection.get(key)
            )

            if detection_timestamp is not None:
                break

    if detection_timestamp is not None:
        plant_detection["timestamp"] = detection_timestamp

    # Riwayat pemupukan.
    fertilization_history = get_data("fertilization_history/latest")

    # Fallback ke node riwayat apabila node /latest tidak ada.
    if not fertilization_history:
        fertilization_history = get_data("fertilization_history")

    latest_fertilization = get_latest_fertilization_record(
        fertilization_history
    )

    days_since_last = calculate_days_since_fertilization(
        latest_fertilization
    )

    fertilization_context = {
        "days_since_last": days_since_last
    }

    # Pertahankan field izin jika catatan terbaru memilikinya.
    if "allowed" in latest_fertilization:
        fertilization_context["allowed"] = (
            latest_fertilization["allowed"]
        )

    context = {
        "sensor": {
            "soil": sensor.get("soil", 0),
            "temperature": sensor.get("temperature", 0),
            "humidity": sensor.get("humidity", 0),
            "ph": sensor.get("ph", 0),
            "rain": sensor.get("rain", False),
        },
        "weather": {
            "forecast": weather.get("forecast", "-"),
            "temperature": weather.get("temperature", 0),
            "humidity": weather.get("humidity", 0),
            "rain_probability": weather.get("rain_probability", 0),
            "rain_3h": weather.get("rain_3h", 0),
        },
        "plant": {
            "plant_count": agent.get("plant_count", 0),
            "age_days": agent.get("age_days", 0),
        },
        "plant_detection": plant_detection,
        "fertilization": fertilization_context,
        "agent": {
            "status": agent.get("status", "-"),
            "pump_duration": agent.get("pump_duration", 0),
        },
    }

    print("\n========== GEMINI CONTEXT DEBUG ==========")
    print("Plant detection:", plant_detection)
    print("Days since fertilization:", days_since_last)
    print("==========================================\n")

    return context


# ============================================================
# FORMAT GEMINI RESULT
# ============================================================

def format_analysis_result(result):
    """Ubah hasil analisis Gemini menjadi teks WhatsApp."""

    if not isinstance(result, dict):
        return str(result)

    action_labels = {
        "NONE": "Belum ada tindakan yang direkomendasikan",
        "IRRIGATE": "Rekomendasi irigasi",
        "FERTILIZE": "Rekomendasi pemupukan",
    }

    action = str(result.get("action", "NONE")).upper()
    action_text = action_labels.get(action, action)

    irrigation_needed = result.get("irrigation_needed")
    fertilization_needed = result.get("fertilization_needed")
    reason = result.get("reason", "-")

    lines = [
        "🌶️ ANALISIS AI LAHAN",
        "",
        f"Rekomendasi: {action_text}",
    ]

    if irrigation_needed is not None:
        lines.append(
            "Kebutuhan irigasi: "
            + ("Ya" if irrigation_needed else "Tidak")
        )

    if fertilization_needed is not None:
        lines.append(
            "Kebutuhan pemupukan: "
            + ("Ya" if fertilization_needed else "Tidak")
        )

    lines.extend([
        "",
        f"Alasan: {reason}",
        "",
        "Catatan: Ini hasil analisis AI, bukan konfirmasi "
        "bahwa pompa atau aktuator sudah dijalankan."
    ])

    return "\n".join(lines)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "service": "smart-agriculture-webhook"}


@app.get("/webhook")
async def health():
    return {"status": "ok"}


# ============================================================
# WHATSAPP WEBHOOK
# ============================================================

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

    # --------------------------------------------------------
    # COMMAND: SET JUMLAH TANAMAN
    # --------------------------------------------------------

    plant_count = parse_set_tanaman(message)

    if plant_count is not None:
        try:
            success = update_data(
                "agent",
                {"plant_count": plant_count}
            )

            if success:
                return send_smart_twiml(
                    f"✅ Jumlah tanaman berhasil diset menjadi "
                    f"{plant_count} tanaman."
                )
            else:
                return send_smart_twiml(
                    "❌ Gagal menyimpan jumlah tanaman ke Firebase."
                )

        except Exception as e:
            print("SET TANAMAN ERROR:", repr(e))
            return send_smart_twiml(
                "❌ Terjadi kesalahan saat menyimpan jumlah tanaman."
            )

        

    # --------------------------------------------------------
    # COMMAND: SET UMUR TANAMAN
    # --------------------------------------------------------

    age_days = parse_set_umur(message)

    if age_days is not None:
        today = datetime.now(
            ZoneInfo("Asia/Jakarta")
        ).date().isoformat()

        try:
            success = update_data(
                "agent",
                {
                    "age_days": age_days,
                    "age_reference_days": age_days,
                    "age_reference_date": today,
                }
            )

            if success:
                return send_smart_twiml(
                    (
                        f"✅ Umur tanaman berhasil diset menjadi "
                        f"{age_days} hari.\n"
                        f"📅 Tanggal acuan: {today}\n"
                        "Umur akan bertambah otomatis setiap "
                        "pergantian hari jika logika pembaruan "
                        "umur di sistem lu aktif."
                    )
                )
            else:
                return send_smart_twiml(
                    "❌ Gagal menyimpan umur tanaman ke Firebase."
                )

        except Exception as e:
            print("SET UMUR ERROR:", repr(e))
            return send_smart_twiml(
                "❌ Terjadi kesalahan saat menyimpan umur tanaman."
            )

        

    # --------------------------------------------------------
    # LOAD DATA FIREBASE DAN CLASSIFY INTENT
    # --------------------------------------------------------

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
            return send_smart_twiml(
                
                "Maaf, perintah belum dapat dikenali."
            )
            

        intent_type = intent.get("intent", "chat")

        # ----------------------------------------------------
        # COMMANDS
        # ----------------------------------------------------

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
                reply = (
                    "☁️ CUACA\n\n"
                    f"Forecast: {weather.get('forecast', '-')}\n"
                    f"Suhu: {weather.get('temperature', 0)} °C\n"
                    f"Humidity: {weather.get('humidity', 0)} %\n"
                    f"Rain Prob: "
                    f"{weather.get('rain_probability', 0)} %"
                )

            elif cmd == "pompa":
                reply = (
                    "🚿 POMPA\n\n"
                    f"Status: {agent.get('status', '-')}\n"
                    f"Durasi: {agent.get('pump_duration', 0)} detik\n"
                    f"Alasan: {agent.get('reason', '-')}"
                )

            elif cmd == "tanaman":
                detected = cam.get(
                    "tanaman",
                    cam.get("plant_detected", False)
                )

                reply = (
                    "🌿 DETEKSI TANAMAN\n\n"
                    f"Terdeteksi: {'Ya' if detected else 'Tidak'}\n"
                    f"Confidence: {cam.get('confidence', 0)}"
                )

            else:
                reply = (
                    "Perintah tidak dikenali.\n"
                    "Coba: status, cuaca, pompa, atau tanaman."
                )

            return send_smart_twiml(reply)

        # ----------------------------------------------------
        # AI ANALYSIS
        # ----------------------------------------------------

        elif intent_type == "analysis":
            try:
                context = build_gemini_context(
                    sensor,
                    weather,
                    agent,
                    cam
                )

                result = analyze_farm(
                    context,
                    purpose="analysis"
                )

                reply = format_analysis_result(result)

            except Exception as e:
                print(
                    f"ANALYSIS ERROR: {type(e).__name__}: {e}"
                )

                reply = (
                    "AI analisis sedang tidak tersedia. "
                    "Silakan coba lagi nanti."
                )
            return send_smart_twiml(reply)
       
        # ----------------------------------------------------
        # AGRICULTURE CHAT
        # ----------------------------------------------------

        else:
            try:
                reply = agriculture_chat_agent(message)

            except Exception as e:
                print(
                    f"CHAT ERROR: {type(e).__name__}: {e}"
                )

                reply = (
                    "AI chat sedang sibuk. "
                    "Silakan coba lagi nanti."
                )

            return send_smart_twiml(reply)

    except Exception as e:
        print(
            f"WEBHOOK FATAL: {type(e).__name__}: {e}"
        )

        # Balas HTTP 200 agar provider tidak terus mengulang
        # pesan hanya karena terjadi error internal handler.
        return send_smart_twiml(
            "Terjadi kesalahan pada sistem. "
            "Silakan coba lagi beberapa saat."
        )

        
