"""ai_orchestrator.py

Orkestrasi sistem AIoT cabai dengan keputusan aktuator deterministik dari
Knowledge Base melalui rule_engine.py. Gemini tidak digunakan untuk memilih
aksi pompa/servo; gemini_service.py tetap dapat digunakan untuk chat/Q&A.

Alur:
Firebase sensor + OpenWeatherMap + YOLO + umur tanaman + riwayat pemupukan
    -> Context Builder -> Knowledge Base Rule Engine
    -> safety-gated decision -> Firebase /aktuator -> ESP32
"""

import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from firebase_service import get_data, update_data
from weather_service import update_weather
from rule_engine import evaluate_rules


# ============================================================
# KONSTANTA
# ============================================================

ACTION_NONE = "NONE"
ACTION_IRRIGATE = "IRRIGATE"
ACTION_FERTILIZE = "FERTILIZE"

SERVO_OFF = "OFF"
SERVO_FERTILIZE = "FERTILIZE"
AI_CYCLE_INTERVAL = 60


# ============================================================
# UTILITAS
# ============================================================

def safe_float(value, default=None):
    """Konversi nilai numerik secara aman; menolak NaN/inf."""
    try:
        if value is None or isinstance(value, bool):
            return default
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError, OverflowError):
        return default


def safe_int(value, default=None):
    """Konversi integer secara aman."""
    number = safe_float(value)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_bool(value, default=None):
    """Konversi boolean Firebase tanpa menganggap data hilang sebagai False."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "ya", "on", "hujan", "rain", "detected"):
            return True
        if normalized in ("false", "0", "no", "tidak", "off", "kering", "none", "not_detected", ""):
            return False
    return default


def calculate_plant_age(agent):
    """Menghitung HST berdasarkan age_reference_date (WIB), bila tersedia."""
    if not isinstance(agent, dict):
        return None

    reference_days = safe_int(agent.get("age_reference_days"))
    reference_date = agent.get("age_reference_date")

    # Kompatibilitas dengan format data lama.
    if reference_days is None or not reference_date:
        return safe_int(agent.get("age_days"))

    try:
        start_date = datetime.strptime(str(reference_date), "%Y-%m-%d").date()
        today = datetime.now(ZoneInfo("Asia/Jakarta")).date()
        elapsed_days = (today - start_date).days
        if elapsed_days < 0:
            return None
        return reference_days + elapsed_days
    except (TypeError, ValueError):
        return safe_int(agent.get("age_days"))


def calculate_fertilization_history(history):
    """Menghitung hari sejak pemupukan; riwayat hilang tetap None (fail-safe)."""
    if not isinstance(history, dict):
        history = {}

    last_applied_at = safe_int(history.get("last_applied_at"))
    last_applied_date = history.get("last_applied_date")
    last_command_id = safe_int(history.get("last_command_id"))
    now = int(time.time())

    days_since_last = None
    history_available = False
    if last_applied_at is not None and 100000 <= last_applied_at <= now:
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
    """Mengambil data terbaru dari Firebase."""
    sensor = get_data("sensor")
    weather = get_data("weather")
    cam = get_data("cam")
    agent = get_data("agent")
    fertilization_history = get_data("fertilization_history/latest")

    plant_age = calculate_plant_age(agent) if isinstance(agent, dict) else None
    planting_date = agent.get("planting_date") if isinstance(agent, dict) else None

    return {
        "sensor": sensor,
        "weather": weather,
        "cam": cam,
        "plant_age": plant_age,
        "planting_date": planting_date,
        "fertilization_history": fertilization_history,
    }


# ============================================================
# MEMBANGUN CONTEXT RULE ENGINE
# ============================================================

def build_ai_context(data):
    """Normalisasi context sesuai kontrak input rule_engine.evaluate_rules().

    Rule Engine membaca soil_moisture, ph, dan rain di level teratas. Data
    sensor bersarang tetap disertakan agar laporan/debugging tetap informatif.
    """
    sensor = data.get("sensor") if isinstance(data.get("sensor"), dict) else {}
    weather = data.get("weather") if isinstance(data.get("weather"), dict) else {}
    cam = data.get("cam") if isinstance(data.get("cam"), dict) else {}

    history = calculate_fertilization_history(data.get("fertilization_history"))

    raw_detection = cam.get("tanaman")
    confidence = safe_float(cam.get("confidence"))
    detection_timestamp = safe_int(cam.get("timestamp"))
    plant_age = safe_int(data.get("plant_age"))

    soil_moisture = safe_float(sensor.get("soil"))
    ph_value = safe_float(sensor.get("ph"))
    local_rain = safe_bool(sensor.get("rain"), default=None)
    rain_probability = safe_float(weather.get("rain_probability"))

    # Field top-level berikut wajib karena dibaca langsung oleh rule_engine.py.
    return {
        "soil_moisture": soil_moisture,
        "ph": ph_value,
        "rain": local_rain,
        "rain_probability": rain_probability,
        "sensor": {
            "soil_moisture": soil_moisture,
            "temperature": safe_float(sensor.get("temperature")),
            "humidity": safe_float(sensor.get("humidity")),
            "ph": ph_value,
            "rain": local_rain,
            "timestamp": safe_int(sensor.get("timestamp")),
        },
        "weather": {
            "forecast": weather.get("forecast"),
            "temperature": safe_float(weather.get("temperature")),
            "humidity": safe_float(weather.get("humidity")),
            "rain_3h": safe_float(weather.get("rain_3h")),
            "rain_probability": rain_probability,
            "timestamp": safe_int(weather.get("timestamp")),
        },
        "plant": {
            "age_days": plant_age,
            "planting_date": data.get("planting_date"),
        },
        "plant_detection": {
            "plant_detected": safe_bool(raw_detection, default=False),
            "raw_detection": raw_detection,
            "confidence": confidence,
            "timestamp": detection_timestamp,
        },
        "fertilization": history,
        "farming_profile": os.getenv("FARMING_PROFILE", "").strip() or None,
    }


# ============================================================
# NORMALISASI HASIL RULE ENGINE
# ============================================================

def normalize_ai_result(result):
    """Validasi bentuk hasil Rule Engine dan pertahankan seluruh metadata."""
    if not isinstance(result, dict):
        return {
            "action": ACTION_NONE,
            "irrigation_needed": False,
            "fertilization_needed": False,
            "analysis": "Hasil Rule Engine tidak valid.",
            "risk": "Data keputusan tidak valid.",
            "recommendation": "Periksa log Rule Engine.",
            "reason": "Hasil Rule Engine tidak valid; aktuator ditahan.",
            "decision_source": "knowledge_base_rule_engine",
            "safety_status": "BLOCKED",
            "blocked_reason": "invalid_rule_engine_result",
        }

    normalized = dict(result)
    action = str(result.get("action", ACTION_NONE)).strip().upper()
    if action not in {ACTION_NONE, ACTION_IRRIGATE, ACTION_FERTILIZE}:
        action = ACTION_NONE

    irrigation_needed = safe_bool(result.get("irrigation_needed"), default=False)
    fertilization_needed = safe_bool(result.get("fertilization_needed"), default=False)

    # Jangan izinkan action yang bertentangan dengan flag kebutuhannya.
    if action == ACTION_IRRIGATE and not irrigation_needed:
        action = ACTION_NONE
        normalized["blocked_reason"] = "inconsistent_irrigation_flag"
        normalized["reason"] = "Perintah irigasi ditahan: irrigation_needed bukan true."
        normalized["safety_status"] = "BLOCKED"
    elif action == ACTION_FERTILIZE and not fertilization_needed:
        action = ACTION_NONE
        normalized["blocked_reason"] = "inconsistent_fertilization_flag"
        normalized["reason"] = "Perintah pemupukan ditahan: fertilization_needed bukan true."
        normalized["safety_status"] = "BLOCKED"

    normalized.update({
        "action": action,
        "irrigation_needed": irrigation_needed,
        "fertilization_needed": fertilization_needed,
        "analysis": result.get("analysis", result.get("reason", "")),
        "risk": result.get("risk", ""),
        "recommendation": result.get("recommendation", ""),
        "reason": str(normalized.get("reason", "")),
        "decision_source": result.get("decision_source", "knowledge_base_rule_engine"),
    })
    return normalized


# ============================================================
# MEMBENTUK COMMAND AKTUATOR
# ============================================================

def build_actuator_command(decision):
    """Membentuk payload dengan nama field yang diharapkan sketch ESP32."""
    action = decision.get("action", ACTION_NONE)
    irrigation_needed = safe_bool(decision.get("irrigation_needed"), False)
    fertilization_needed = safe_bool(decision.get("fertilization_needed"), False)

    common = {
        "timestamp": int(time.time()),
        "irrigation_needed": irrigation_needed,
        "fertilization_needed": fertilization_needed,
        "reason": str(decision.get("reason", "")),
    }

    if action == ACTION_IRRIGATE and irrigation_needed:
        return {
            **common,
            "pompa": True,
            "servo": SERVO_OFF,
            "pump_duration": 20,
            "action": ACTION_IRRIGATE,
        }

    if action == ACTION_FERTILIZE and fertilization_needed:
        return {
            **common,
            "pompa": False,
            "servo": SERVO_FERTILIZE,
            "pump_duration": 0,
            "action": ACTION_FERTILIZE,
        }

    return {
        **common,
        "pompa": False,
        "servo": SERVO_OFF,
        "pump_duration": 0,
        "action": ACTION_NONE,
    }


# ============================================================
# PEMERIKSAAN STATUS AKTUATOR
# ============================================================

def check_actuator_ready():
    """Hanya mengirim perintah baru saat ESP32 terdeteksi idle."""
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
        return False, status, current_command, "ID perintah tidak tersedia atau tidak valid."

    if command_id > last_executed:
        return False, status, current_command, "Masih ada perintah yang belum diproses ESP32."

    if active_command_id not in (None, 0):
        return False, status, current_command, "Aktuator masih memiliki perintah aktif."

    if str(status.get("action", ACTION_NONE)).upper().strip() != ACTION_NONE:
        return False, status, current_command, "Status aktuator belum idle."
    if safe_bool(status.get("pompa"), default=False):
        return False, status, current_command, "Pompa masih aktif."
    if str(status.get("servo", SERVO_OFF)).upper().strip() != SERVO_OFF:
        return False, status, current_command, "Servo masih aktif."

    return True, status, current_command, "Aktuator siap."


def generate_command_id(status, current_command):
    """Membuat ID epoch yang lebih besar dari ID yang sudah diketahui."""
    now = int(time.time())
    last_executed = safe_int(status.get("last_executed_command"), 0)
    previous_id = safe_int(current_command.get("timestamp"), 0)
    return max(now, last_executed + 1, previous_id + 1)


# ============================================================
# MENERBITKAN COMMAND AKTUATOR
# ============================================================

def publish_actuator_command(decision):
    """Publikasikan aksi valid satu kali sampai keputusan kembali NONE."""
    action = decision.get("action", ACTION_NONE)
    control = get_data("ai_control")
    if not isinstance(control, dict):
        control = {}

    last_published_action = str(
        control.get("last_published_action", ACTION_NONE)
    ).upper().strip()

    if action == ACTION_NONE:
        reset_ok = update_data("ai_control", {"last_published_action": ACTION_NONE})
        if not reset_ok:
            print("[ACTUATOR] WARNING: latch NONE gagal disimpan.")
        return False, "Tidak ada tindakan yang perlu diterbitkan."

    if action not in {ACTION_IRRIGATE, ACTION_FERTILIZE}:
        return False, "Tindakan tidak valid."
    if action == ACTION_IRRIGATE and not decision.get("irrigation_needed"):
        return False, "Kebutuhan irigasi tidak terpenuhi."
    if action == ACTION_FERTILIZE and not decision.get("fertilization_needed"):
        return False, "Kebutuhan pemupukan tidak terpenuhi."
    if last_published_action == action:
        return False, f"Tindakan {action} sudah diterbitkan; menunggu keputusan kembali NONE."

    ready, status, current_command, message = check_actuator_ready()
    if not ready:
        print("[ACTUATOR] Command tidak diterbitkan:", message)
        return False, message

    command_id = generate_command_id(status, current_command)
    command = build_actuator_command(decision)
    command["timestamp"] = command_id

    if not update_data("aktuator", command):
        return False, "Gagal menyimpan perintah ke Firebase."

    latch_saved = update_data("ai_control", {
        "last_published_action": action,
        "last_published_command_id": command_id,
    })
    if not latch_saved:
        print("[ACTUATOR] WARNING: command terkirim, tetapi latch gagal disimpan.")

    print(f"[ACTUATOR] Perintah {action} diterbitkan. Command ID: {command_id}")
    return True, f"Perintah {action} berhasil diterbitkan."


# ============================================================
# MENYIMPAN HASIL ANALISIS
# ============================================================

def save_results(decision, actuator_report, context, published_action):
    """Simpan hasil keputusan ke /ai dan metadata tanaman ke /agent."""
    timestamp = int(time.time())
    ai_data = {
        "analysis": decision.get("analysis"),
        "risk": decision.get("risk"),
        "recommendation": decision.get("recommendation"),
        "irrigation_needed": bool(decision.get("irrigation_needed", False)),
        "fertilization_needed": bool(decision.get("fertilization_needed", False)),
        "action_recommended": decision.get("action", ACTION_NONE),
        "action_published": published_action,
        "reason": decision.get("reason"),
        "decision_source": decision.get("decision_source", "knowledge_base_rule_engine"),
        "kb_source": decision.get("kb_source"),
        "safety_status": decision.get("safety_status"),
        "blocked_reason": decision.get("blocked_reason"),
        "timestamp": timestamp,
    }
    update_data("ai", ai_data)

    plant = context.get("plant", {})
    detection = context.get("plant_detection", {})
    agent_data = {
        "age_days": plant.get("age_days"),
        "planting_date": plant.get("planting_date"),
        "plant_detected": safe_bool(detection.get("plant_detected"), False),
        "plant_confidence": detection.get("confidence"),
        "recommendation": decision.get("recommendation"),
        "reason": decision.get("reason"),
        "action": decision.get("action", ACTION_NONE),
        "decision_source": decision.get("decision_source", "knowledge_base_rule_engine"),
        "timestamp": timestamp,
    }
    update_data("agent", agent_data)

    print("\n[FIREBASE] Hasil Rule Engine disimpan.")
    print(json.dumps({
        "actuator_report": actuator_report,
        "ai": ai_data,
        "agent": agent_data,
    }, indent=2, ensure_ascii=False))


# ============================================================
# SATU SIKLUS RULE ENGINE
# ============================================================

def run_ai_cycle():
    """Jalankan Rule Engine tanpa ketergantungan pada kuota/API Gemini."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE-BASED RULE ENGINE CYCLE")
    print("=" * 60)

    print("\n[1] Memperbarui cuaca...")
    try:
        weather_result = update_weather()
        if weather_result is None:
            print("[WEATHER] WARNING: update cuaca gagal; data Firebase yang ada akan diperiksa.")
    except Exception as exc:
        print(f"[WEATHER] WARNING: update cuaca gagal: {exc}")

    print("\n[2] Mengambil data sistem...")
    data = collect_system_data()

    print("\n[3] Membangun context...")
    context = build_ai_context(data)
    print(json.dumps(context, indent=2, ensure_ascii=False))

    print("\n[4] Mengevaluasi aturan Knowledge Base...")
    try:
        decision = normalize_ai_result(evaluate_rules(context))
    except Exception as exc:
        print("[RULE ENGINE] ERROR:", exc)
        decision = normalize_ai_result({
            "action": ACTION_NONE,
            "irrigation_needed": False,
            "fertilization_needed": False,
            "reason": f"Rule Engine gagal: {exc}; aktuator ditahan.",
            "analysis": "Keputusan otomatis dihentikan karena terjadi kesalahan Rule Engine.",
            "risk": "Rule Engine error.",
            "recommendation": "Periksa log dan konfigurasi Knowledge Base.",
            "decision_source": "knowledge_base_rule_engine",
            "safety_status": "BLOCKED",
            "blocked_reason": "rule_engine_exception",
        })
        # Fail closed: jangan menerbitkan command pada siklus yang error.
        actuator_report = build_actuator_command(decision)
        actuator_report.update({"action": ACTION_NONE, "pompa": False, "servo": SERVO_OFF, "pump_duration": 0})
        save_results(decision, actuator_report, context, ACTION_NONE)
        return {
            "context": context,
            "ai_result": decision,
            "actuator_command": actuator_report,
            "published": False,
            "publish_message": "Penerbitan dibatalkan karena Rule Engine gagal.",
        }

    print("\n[5] HASIL RULE ENGINE:")
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    print("\n[6] Menerbitkan perintah aktuator jika diizinkan...")
    published, publish_message = publish_actuator_command(decision)
    print("Command published:", published)
    print("Publish status:", publish_message)

    actuator_report = build_actuator_command(decision)
    if not published:
        # Laporan ini merepresentasikan hasil publikasi pada siklus ini,
        # bukan mengubah /aktuator yang mungkin masih berisi command lama.
        actuator_report.update({
            "action": ACTION_NONE,
            "pompa": False,
            "servo": SERVO_OFF,
            "pump_duration": 0,
        })

    print("\n[7] Menyimpan hasil analisis...")
    save_results(
        decision,
        actuator_report,
        context,
        published_action=decision.get("action", ACTION_NONE) if published else ACTION_NONE,
    )

    print("\nSiklus Rule Engine selesai.")
    return {
        "context": context,
        "ai_result": decision,
        "actuator_command": actuator_report,
        "published": published,
        "publish_message": publish_message,
    }


# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    print(
        "AI Orchestrator tidak menjalankan loop otomatis. "
        "Jalankan API dengan: "
        "uvicorn orchestrator_api:app --host 0.0.0.0 --port $PORT"
    )
