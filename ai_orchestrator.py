"""
ai_orchestrator.py

Orkestrasi utama Knowledge-Based AI Agent untuk sistem
pertanian cerdas berbasis AIoT.

Alur:
Firebase Sensor + OpenWeatherMap + ESP32-CAM / YOLO + Plant Age
    -> Context Builder
    -> Knowledge-Based AI Agent (gemini_service.py)
    -> Validated Decision
    -> Firebase /aktuator
    -> ESP32 sebagai final actuator executor

Prinsip:
1. Knowledge-Based AI Agent menentukan kebutuhan agronomis.
2. gemini_service.py menjadi pusat analisis dan safety validation.
3. ai_orchestrator.py tidak membuat aturan agronomi baru.
4. irrigation_needed dan fertilization_needed berasal dari AI Agent.
5. action menunjukkan apakah aktuator boleh dijalankan.
6. ESP32 tetap menjadi eksekutor akhir aktuator.
7. Safety gate agronomis tetap berada di gemini_service.py.
"""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from firebase_service import get_data, update_data
from weather_service import update_weather
from gemini_service import analyze_farm


# ============================================================
# KONSTANTA
# ============================================================

ACTION_NONE = "NONE"
ACTION_IRRIGATE = "IRRIGATE"
ACTION_FERTILIZE = "FERTILIZE"

SERVO_OFF = "OFF"
SERVO_FERTILIZE = "FERTILIZE"

AI_CYCLE_INTERVAL = 30


# ============================================================
# UTILITAS
# ============================================================

def safe_float(value, default=None):
    """Konversi nilai menjadi float dengan aman."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    """Konversi nilai menjadi integer dengan aman."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_plant_age(agent):
    """Menghitung umur tanaman berdasarkan tanggal acuan WIB."""
    if not isinstance(agent, dict):
        return None

    reference_days = safe_int(
        agent.get("age_reference_days")
    )
    reference_date = agent.get("age_reference_date")

    # Kompatibilitas dengan data Firebase lama
    if reference_days is None or not reference_date:
        return safe_int(agent.get("age_days"))

    try:
        start_date = datetime.strptime(
            str(reference_date), "%Y-%m-%d"
        ).date()

        today = datetime.now(
            ZoneInfo("Asia/Jakarta")
        ).date()

        elapsed_days = (today - start_date).days

        return reference_days + max(0, elapsed_days)

    except (TypeError, ValueError):
        return safe_int(agent.get("age_days"))


def safe_bool(value, default=False):
    """Konversi boolean dari Firebase atau hasil AI secara aman."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        value = value.strip().lower()

        if value in ("true", "1", "yes", "ya", "on"):
            return True

        if value in ("false", "0", "no", "tidak", "off", ""):
            return False

    return default


def calculate_fertilization_history(history):
    """
    Menghitung waktu sejak pemupukan terakhir.
    days_since_last None jika riwayat tidak valid/tidak tersedia.
    """
    if not isinstance(history, dict):
        history = {}

    last_applied_at = safe_int(history.get("last_applied_at"))
    last_applied_date = history.get("last_applied_date")
    last_command_id = safe_int(history.get("last_command_id"))

    days_since_last = None
    history_available = False
    now = int(time.time())

    if (
        last_applied_at is not None
        and 0 < last_applied_at <= now
    ):
        days_since_last = (now - last_applied_at) / 86400.0
        history_available = True

    return {
        "last_applied_at": last_applied_at,
        "last_applied_date": last_applied_date,
        "last_command_id": last_command_id,
        "days_since_last": days_since_last,
        "history_available": history_available,
    }


# ============================================================
# PENGAMBILAN DATA SISTEM
# ============================================================

def collect_system_data():
    """Mengambil data sensor, cuaca, kamera, tanaman, dan pemupukan."""
    sensor = get_data("sensor")
    weather = get_data("weather")
    cam = get_data("cam")
    agent = get_data("agent")
    fertilization_history = get_data("fertilization_history/latest")

    plant_age = None
    planting_date = None

    if isinstance(agent, dict):
        plant_age = calculate_plant_age(agent)
        planting_date = agent.get("planting_date")

    return {
        "sensor": sensor,
        "weather": weather,
        "cam": cam,
        "plant_age": plant_age,
        "planting_date": planting_date,
        "fertilization_history": fertilization_history,
    }


# ============================================================
# MEMBANGUN CONTEXT AI
# ============================================================

def build_ai_context(data):
    """
    Normalisasi data ke bentuk context yang digunakan oleh
    gemini_service.py. Keputusan agronomis dan safety validation
    tetap dilakukan oleh gemini_service.py.
    """
    sensor = data.get("sensor")
    weather = data.get("weather")
    cam = data.get("cam")
    plant_age = data.get("plant_age")
    planting_date = data.get("planting_date")
    fertilization_history = data.get("fertilization_history")

    if not isinstance(sensor, dict):
        sensor = {}
    if not isinstance(weather, dict):
        weather = {}
    if not isinstance(cam, dict):
        cam = {}

    history = calculate_fertilization_history(fertilization_history)

    raw_detection = cam.get("tanaman")
    confidence = safe_float(cam.get("confidence"))
    detection_timestamp = safe_int(cam.get("timestamp"))
    age_days = safe_int(plant_age)

    return {
        "sensor": {
            "soil_moisture": safe_float(sensor.get("soil")),
            "temperature": safe_float(sensor.get("temperature")),
            "humidity": safe_float(sensor.get("humidity")),
            "ph": safe_float(sensor.get("ph")),
            "rain": safe_bool(sensor.get("rain"), default=False),
        },
        "weather": {
            "forecast": weather.get("forecast"),
            "temperature": safe_float(weather.get("temperature")),
            "humidity": safe_float(weather.get("humidity")),
            "rain_3h": safe_float(weather.get("rain_3h")),
            "rain_probability": safe_float(weather.get("rain_probability")),
        },
        "plant": {
            "age_days": age_days,
            "planting_date": planting_date,
        },
        "plant_detection": {
            "plant_detected": safe_bool(raw_detection, default=False),
            "raw_detection": raw_detection,
            "confidence": confidence,
            "timestamp": detection_timestamp,
        },
        "fertilization": history,
    }


# ============================================================
# NORMALISASI HASIL AI
# ============================================================

def normalize_ai_result(ai_result):
    """Memastikan struktur hasil AI valid tanpa menambah aturan agronomi."""
    if not isinstance(ai_result, dict):
        return {
            "action": ACTION_NONE,
            "irrigation_needed": False,
            "fertilization_needed": False,
            "analysis": None,
            "risk": "Respons AI tidak valid.",
            "recommendation": None,
            "reason": "Respons AI tidak valid.",
        }

    action = str(ai_result.get("action", ACTION_NONE)).upper().strip()
    if action not in {ACTION_NONE, ACTION_IRRIGATE, ACTION_FERTILIZE}:
        action = ACTION_NONE

    return {
        "action": action,
        "irrigation_needed": safe_bool(
            ai_result.get("irrigation_needed"), default=False
        ),
        "fertilization_needed": safe_bool(
            ai_result.get("fertilization_needed"), default=False
        ),
        "analysis": ai_result.get("analysis"),
        "risk": ai_result.get("risk"),
        "recommendation": ai_result.get("recommendation"),
        "reason": str(ai_result.get("reason", "")),
    }


# ============================================================
# MEMBENTUK COMMAND AKTUATOR
# ============================================================

def build_actuator_command(ai_result):
    """Membentuk payload command berdasarkan keputusan tervalidasi."""
    action = ai_result.get("action", ACTION_NONE)

    irrigation_needed = safe_bool(
        ai_result.get("irrigation_needed"), default=False
    )
    fertilization_needed = safe_bool(
        ai_result.get("fertilization_needed"), default=False
    )

    if action == ACTION_IRRIGATE:
        return {
            "pompa": True,
            "servo": SERVO_OFF,
            "pump_duration": 20,
            "action": ACTION_IRRIGATE,
            "timestamp": int(time.time()),
            "irrigation_needed": irrigation_needed,
            "fertilization_needed": fertilization_needed,
        }

    if action == ACTION_FERTILIZE:
        return {
            "pompa": False,
            "servo": SERVO_FERTILIZE,
            "pump_duration": 0,
            "action": ACTION_FERTILIZE,
            "timestamp": int(time.time()),
            "irrigation_needed": irrigation_needed,
            "fertilization_needed": fertilization_needed,
        }

    return {
        "pompa": False,
        "servo": SERVO_OFF,
        "pump_duration": 0,
        "action": ACTION_NONE,
        "timestamp": int(time.time()),
        "irrigation_needed": irrigation_needed,
        "fertilization_needed": fertilization_needed,
    }


# ============================================================
# PEMERIKSAAN STATUS AKTUATOR
# ============================================================

def check_actuator_ready():
    """
    Mengirim command hanya jika status ESP32 diketahui dan idle,
    serta tidak ada command lama yang masih menunggu diproses.
    """
    status = get_data("aktuator_status")
    current_command = get_data("aktuator")

    if not isinstance(status, dict):
        return False, None, None, "Status aktuator tidak tersedia."

    if not isinstance(current_command, dict):
        return False, status, None, "Data perintah aktuator tidak tersedia."

    last_executed = safe_int(status.get("last_executed_command"))
    active_command_id = safe_int(status.get("active_command_id"))
    command_id = safe_int(current_command.get("timestamp"))

    if last_executed is None or command_id is None:
        return (
            False, status, current_command,
            "ID perintah tidak tersedia atau tidak valid."
        )

    if command_id > last_executed:
        return (
            False, status, current_command,
            "Masih ada perintah yang belum diproses ESP32."
        )

    if active_command_id not in (None, 0):
        return False, status, current_command, "Aktuator masih memiliki perintah aktif."

    status_action = str(status.get("action", "NONE")).upper().strip()
    if status_action != ACTION_NONE:
        return False, status, current_command, "Status aktuator belum idle."

    if safe_bool(status.get("pompa"), default=False):
        return False, status, current_command, "Pompa masih aktif."

    if str(status.get("servo", "OFF")).upper().strip() != SERVO_OFF:
        return False, status, current_command, "Servo masih aktif."

    return True, status, current_command, "Aktuator siap."


def generate_command_id(status, current_command):
    """Membuat command ID yang lebih besar dari ID sebelumnya."""
    now = int(time.time())
    last_executed = safe_int(status.get("last_executed_command"), default=0)
    previous_id = safe_int(current_command.get("timestamp"), default=0)

    return max(now, last_executed + 1, previous_id + 1)


# ============================================================
# MENERBITKAN COMMAND AKTUATOR
# ============================================================

def publish_actuator_command(ai_result):
    """
    Menerbitkan command jika:
    - keputusan AI valid dan konsisten dengan flag kebutuhan;
    - aksi yang sama belum diterbitkan sejak keputusan terakhir NONE;
    - ESP32 melaporkan status idle.
    """
    action = ai_result.get("action", ACTION_NONE)

    control = get_data("ai_control")
    if not isinstance(control, dict):
        control = {}

    last_published_action = str(
        control.get("last_published_action", ACTION_NONE)
    ).upper().strip()

    # NONE hanya mereset latch jika ini hasil AI yang sah.
    if action == ACTION_NONE:
        reset_ok = update_data(
            "ai_control",
            {"last_published_action": ACTION_NONE}
        )
        if not reset_ok:
            print("[ACTUATOR] WARNING: latch NONE gagal disimpan.")
        return False, "Tidak ada tindakan yang perlu diterbitkan."

    if action not in {ACTION_IRRIGATE, ACTION_FERTILIZE}:
        return False, "Tindakan tidak valid."

    if action == ACTION_IRRIGATE and not safe_bool(
        ai_result.get("irrigation_needed"), default=False
    ):
        return False, "Kebutuhan irigasi tidak terpenuhi."

    if action == ACTION_FERTILIZE and not safe_bool(
        ai_result.get("fertilization_needed"), default=False
    ):
        return False, "Kebutuhan pemupukan tidak terpenuhi."

    if last_published_action == action:
        return (
            False,
            f"Tindakan {action} sudah diterbitkan; "
            "menunggu keputusan AI kembali ke NONE."
        )

    ready, status, current_command, reason = check_actuator_ready()
    if not ready:
        print("[ACTUATOR] Command tidak diterbitkan:", reason)
        return False, reason

    command_id = generate_command_id(status, current_command)
    command = build_actuator_command(ai_result)
    command["timestamp"] = command_id
    command["reason"] = ai_result.get("reason", "")

    # Tulis command ke node /aktuator; jangan menulis NONE setiap siklus.
    if not update_data("aktuator", command):
        return False, "Gagal menyimpan perintah ke Firebase."

    # Catat latch setelah command berhasil dikirim.
    if not update_data(
        "ai_control",
        {
            "last_published_action": action,
            "last_published_command_id": command_id,
        },
    ):
        print(
            "[ACTUATOR] WARNING: command sudah dikirim, "
            "tetapi latch gagal disimpan. Periksa Firebase."
        )

    print(
        f"[ACTUATOR] Perintah {action} diterbitkan. "
        f"Command ID: {command_id}"
    )
    return True, f"Perintah {action} berhasil diterbitkan."


# ============================================================
# MENYIMPAN HASIL ANALISIS
# ============================================================

def save_results(ai_result, actuator_report, context, published_action):
    """
    Menyimpan laporan analisis ke /ai dan /agent.
    Fungsi ini TIDAK menulis ke /aktuator.
    """
    timestamp = int(time.time())

    ai_data = {
        "analysis": ai_result.get("analysis"),
        "risk": ai_result.get("risk"),
        "recommendation": ai_result.get("recommendation"),
        "irrigation_needed": safe_bool(
            ai_result.get("irrigation_needed"), default=False
        ),
        "fertilization_needed": safe_bool(
            ai_result.get("fertilization_needed"), default=False
        ),
        "action_recommended": ai_result.get("action", ACTION_NONE),
        "action_published": published_action,
        "reason": ai_result.get("reason"),
        "timestamp": timestamp,
    }

    update_data("ai", ai_data)

    plant = context.get("plant", {})
    plant_detection = context.get("plant_detection", {})

    agent_data = {
        "age_days": plant.get("age_days"),
        "planting_date": plant.get("planting_date"),
        "plant_detected": safe_bool(
            plant_detection.get("plant_detected"), default=False
        ),
        "recommendation": ai_result.get("recommendation"),
        "reason": ai_result.get("reason"),
        "action": ai_result.get("action", ACTION_NONE),
        "timestamp": timestamp,
    }

    update_data("agent", agent_data)

    print("\n[FIREBASE] Analysis results saved.")
    print(
        json.dumps(
            {
                "actuator_report": actuator_report,
                "ai": ai_data,
                "agent": agent_data,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================
# SATU SIKLUS AI
# ============================================================

def run_ai_cycle():
    """Menjalankan satu siklus analisis, publikasi, dan pencatatan."""
    print()
    print("=" * 60)
    print("AI AGENT CYCLE")
    print("=" * 60)

    print("\n[1] Updating weather...")
    weather = update_weather()
    if weather is None:
        print("WARNING: Weather update gagal.")

    print("\n[2] Collecting system data...")
    data = collect_system_data()

    print("\n[3] Building AI context...")
    context = build_ai_context(data)
    print(json.dumps(context, indent=2, ensure_ascii=False))

    print("\n[4] Calling Knowledge-Based AI Agent...")
    ai_success = True

    try:
        ai_result = analyze_farm(context)
    except Exception as exc:
        ai_success = False
        print("Gemini ERROR:", exc)
        ai_result = {
            "action": ACTION_NONE,
            "irrigation_needed": False,
            "fertilization_needed": False,
            "analysis": None,
            "risk": "AI Agent gagal memberikan keputusan.",
            "recommendation": None,
            "reason": (
                "Aktuator tidak dijalankan karena "
                "Knowledge-Based AI Agent gagal."
            ),
        }

    ai_result = normalize_ai_result(ai_result)

    print("\n[5] AI RESULT:")
    print(json.dumps(ai_result, indent=2, ensure_ascii=False))

    print("\n[6] Publishing actuator command...")
    if ai_success:
        published, publish_message = publish_actuator_command(ai_result)
    else:
        # Jangan reset latch ketika AI gagal; hasil fallback NONE
        # bukan keputusan agronomis yang valid.
        published = False
        publish_message = "Penerbitan dilewati karena AI gagal."

    print("Command published:", published)
    print("Publish status:", publish_message)

    # Laporan ini tidak pernah ditulis ke /aktuator.
    actuator_report = build_actuator_command(ai_result)
    if not published:
        actuator_report["action"] = ACTION_NONE
        actuator_report["pompa"] = False
        actuator_report["servo"] = SERVO_OFF
        actuator_report["pump_duration"] = 0

    print("\n[7] Saving analysis results...")
    save_results(
        ai_result,
        actuator_report,
        context,
        published_action=(
            ai_result.get("action", ACTION_NONE)
            if published
            else ACTION_NONE
        ),
    )

    print("\nAI cycle selesai.")
    return {
        "context": context,
        "ai_result": ai_result,
        "actuator_command": actuator_report,
        "published": published,
        "publish_message": publish_message,
    }


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 60)
    print("KNOWLEDGE-BASED AI AGENT")
    print("SMART AGRICULTURE - CHILI")
    print("=" * 60)
    print(f"\nAI cycle interval: {AI_CYCLE_INTERVAL} detik")
    print("\nSafety validation:")
    print("- YOLO        : gemini_service.py")
    print("- Local rain  : gemini_service.py")
    print("- OWM rain    : gemini_service.py")
    print("- pH          : gemini_service.py")
    print("- Fertilizer cooldown : gemini_service.py")
    print("- Actuator executor   : ESP32")

    while True:
        try:
            run_ai_cycle()
        except KeyboardInterrupt:
            print("\nAI Agent dihentikan.")
            break
        except Exception as exc:
            print("\nUNEXPECTED ERROR:", exc)

        print(f"\nMenunggu {AI_CYCLE_INTERVAL} detik...")
        time.sleep(AI_CYCLE_INTERVAL)


if __name__ == "__main__":
    main()
