"""Deterministic, Knowledge-Base-driven actuator decision engine.

Gemini is intentionally not used here.  This module reads agronomic rules from
Firebase/local KnowledgeBase and returns a command compatible with the current
ESP32 /aktuator schema.

Expected context (built by ai_orchestrator.build_ai_context):
  soil_moisture: 0..100 (%)
  ph: float
  rain: bool | None                  # local rain sensor state
  weather.rain_probability: 0..100  # OWM forecast probability
  plant.age_days: int
  plant_detection: {plant_detected, confidence, timestamp}
  fertilization: {days_since_last, last_applied_at, ...}

Fail-safe policy: missing/invalid essential knowledge or safety data prevents
actuation. Agronomic need flags are kept distinct from the final safety-gated
'action'.
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    from knowledge_base import get_knowledge_base
except Exception:  # pragma: no cover - handled fail-safe by evaluate_rules
    get_knowledge_base = None  # type: ignore[assignment]

VALID_ACTIONS = {"NONE", "IRRIGATE", "FERTILIZE"}
MIN_YOLO_CONFIDENCE = 0.25
MAX_YOLO_AGE_SECONDS = 300
OWM_RAIN_PROBABILITY_THRESHOLD = 70.0
# KB says 10–14 days. Use the earliest permitted day (10) as the due threshold.
DEFAULT_NPK_MIN_INTERVAL_DAYS = 10
DEFAULT_FERTILIZATION_START_AGE_DAYS = 30


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_number(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None:
        return None
    return int(number)


def _bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "ya", "hujan", "rain", "detected"}:
            return True
        if normalized in {"false", "0", "no", "tidak", "kering", "none", "off", "not_detected"}:
            return False
    return None


def _get_path(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _kb_topics(kb: Any) -> Dict[str, Any]:
    """Accept either a KnowledgeBase object with load() or an already-loaded dict."""
    if isinstance(kb, dict):
        # get_knowledge_base() may already return the Firebase data as a dict.
        loaded = kb
    elif callable(getattr(kb, "load", None)):
        loaded = kb.load()
    else:
        raise TypeError(
            "get_knowledge_base() harus mengembalikan dictionary atau objek KnowledgeBase dengan load()"
        )

    if not isinstance(loaded, dict):
        raise ValueError("Data Knowledge Base yang dimuat bukan dictionary")

    # Support both a direct root dictionary and a wrapped {"knowledge_base": {...}}.
    if isinstance(loaded.get("knowledge_base"), dict):
        loaded = loaded["knowledge_base"]
    return loaded


def _extract_ph_tolerance(soil_topic: Any) -> Tuple[Optional[float], Optional[float]]:
    """Handle the currently described soil.ph.tolerable layout and variants."""
    candidates = [
        _get_path(soil_topic, "ph", "tolerable"),
        _get_path(soil_topic, "tolerable"),
        _get_path(soil_topic, "ph", "range"),
        _get_path(soil_topic, "ph"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            low = _number(candidate.get("min"))
            high = _number(candidate.get("max"))
            if low is not None and high is not None and low < high:
                return low, high
    return None, None


def _extract_moisture_limits(irrigation_topic: Any) -> Tuple[Optional[float], Optional[float]]:
    ideal = _get_path(irrigation_topic, "soil_moisture", "ideal")
    if not isinstance(ideal, dict):
        return None, None
    low = _number(ideal.get("min"))
    high = _number(ideal.get("max"))
    if low is None or high is None or low < 0 or high > 100 or low >= high:
        return None, None
    return low, high


def _extract_npk_schedule(fertilization_topic: Any, profile: str) -> Tuple[Optional[int], Optional[int], str]:
    """Return (start-age-days, minimum-interval-days, explanation).

    The current KB example uses e.g. supplementary.dryland_highland_medium.
    NPK application strings can be '10-14_hari_sekali_mulai_umur_1_bulan' or
    similar Indonesian text. Unsupported/missing profiles fail closed for
    fertilization, but do not disable otherwise-valid irrigation decisions.
    """
    if not profile:
        return None, None, "FARMING_PROFILE belum dikonfigurasi; aturan pemupukan tidak dapat dipastikan."

    supplementary = _get_path(fertilization_topic, "supplementary")
    if not isinstance(supplementary, dict):
        return None, None, "Knowledge Base tidak memiliki bagian fertilization.supplementary."

    profile_data = supplementary.get(profile)
    if not isinstance(profile_data, dict):
        return None, None, f"Profil budidaya '{profile}' tidak ditemukan di Knowledge Base."

    # Prefer the explicit NPK 16-16-16 item used by the current KB.
    npk = profile_data.get("NPK_16-16-16")
    if not isinstance(npk, dict):
        # tolerate differences in key formatting, but only choose an NPK item
        npk = next((v for k, v in profile_data.items()
                    if "npk" in str(k).lower() and isinstance(v, dict)), None)
    if not isinstance(npk, dict):
        return None, None, f"Aturan NPK tidak ditemukan pada profil '{profile}'."

    application = str(npk.get("application", "")).strip().lower()
    # Interval: match the range in phrases such as '10-14_hari_sekali'.
    interval_match = re.search(r"(\d+)\s*[-– sampai]+\s*(\d+)\s*[_ ]*hari", application)
    interval_days: Optional[int] = None
    if interval_match:
        interval_days = int(interval_match.group(1))
    else:
        simple = re.search(r"(\d+)\s*[_ ]*hari\s*sekali", application)
        if simple:
            interval_days = int(simple.group(1))

    # The application string says 'mulai_umur_1_bulan'; interpret as 30 HST.
    start_age_days = DEFAULT_FERTILIZATION_START_AGE_DAYS if (
        "umur_1_bulan" in application or "umur 1 bulan" in application or "mulai_umur_satu_bulan" in application
    ) else None

    if start_age_days is None:
        # If KB uses a numeric explicit age, support it. Do not invent one when
        # there is no age rule available.
        age_match = re.search(r"(?:umur|usia|mulai)[_ ]*(\d+)\s*[_ ]*(?:hari|hst)", application)
        if age_match:
            start_age_days = int(age_match.group(1))

    if interval_days is None or interval_days <= 0:
        return None, None, f"Jadwal interval NPK tidak dapat dibaca dari: '{application}'."
    if start_age_days is None:
        return None, None, f"Umur mulai pemupukan tidak dapat dibaca dari: '{application}'."
    return start_age_days, interval_days, f"Jadwal NPK dari KB: mulai umur {start_age_days} HST, interval minimal {interval_days} hari."


def _yolo_status(plant_detection: Any, now: Optional[float] = None) -> Tuple[bool, str]:
    if not isinstance(plant_detection, dict):
        return False, "Data deteksi YOLO tidak tersedia."

    detected = _bool_value(plant_detection.get("plant_detected"))
    confidence = _number(plant_detection.get("confidence"))
    timestamp = _number(plant_detection.get("timestamp"))

    if detected is not True:
        return False, "YOLO tidak mengonfirmasi keberadaan tanaman."
    if confidence is None or confidence < MIN_YOLO_CONFIDENCE:
        return False, f"Confidence YOLO tidak valid atau di bawah {MIN_YOLO_CONFIDENCE:.2f}."
    if timestamp is None or timestamp <= 0:
        return False, "Timestamp deteksi YOLO tidak tersedia/invalid."

    # Accept epoch seconds or milliseconds if the upstream service uses ms.
    if timestamp > 100_000_000_000:
        timestamp /= 1000.0
    current = time.time() if now is None else now
    age = current - timestamp
    if age < -30:
        return False, "Timestamp YOLO berada terlalu jauh di masa depan."
    if age > MAX_YOLO_AGE_SECONDS:
        return False, f"Data YOLO kedaluwarsa ({age:.0f} detik; batas {MAX_YOLO_AGE_SECONDS} detik)."
    return True, "YOLO valid."


def _rain_probability(context: Dict[str, Any]) -> Optional[float]:
    weather = context.get("weather")
    value = None
    if isinstance(weather, dict):
        value = weather.get("rain_probability")
    if value is None:
        value = context.get("rain_probability")
    probability = _number(value)
    if probability is None or probability < 0 or probability > 100:
        return None
    return probability


def _fertilization_days_since(context: Dict[str, Any]) -> Optional[float]:
    history = context.get("fertilization")
    if not isinstance(history, dict):
        history = context.get("fertilization_history")
    if not isinstance(history, dict):
        return None
    return _number(history.get("days_since_last"))


def _plant_age(context: Dict[str, Any]) -> Optional[int]:
    plant = context.get("plant")
    value = plant.get("age_days") if isinstance(plant, dict) else None
    if value is None:
        value = context.get("plant_age_days", context.get("age_days"))
    age = _int_number(value)
    return age if age is not None and age >= 0 else None


def _result(
    action: str = "NONE",
    irrigation_needed: bool = False,
    fertilization_needed: bool = False,
    reason: str = "",
    analysis: str = "",
    risk: str = "",
    recommendation: str = "",
    kb_source: str = "unknown",
    safety_status: str = "BLOCKED",
    blocked_reason: str = "",
) -> Dict[str, Any]:
    if action not in VALID_ACTIONS:
        action = "NONE"
    return {
        "action": action,
        "irrigation_needed": bool(irrigation_needed),
        "fertilization_needed": bool(fertilization_needed),
        "analysis": analysis or reason,
        "risk": risk,
        "recommendation": recommendation,
        "reason": reason or "Tidak ada tindakan otomatis yang memenuhi aturan.",
        "decision_source": "knowledge_base_rule_engine",
        "kb_source": kb_source,
        "safety_status": safety_status,
        "blocked_reason": blocked_reason,
    }


def evaluate_rules(context: Dict[str, Any], kb: Any = None, now: Optional[float] = None) -> Dict[str, Any]:
    """Evaluate KB-derived decisions and return a safety-gated actuator action."""
    if not isinstance(context, dict):
        return _result(reason="Konteks sensor tidak valid; aktuator ditahan.",
                       risk="Data input tidak valid.", blocked_reason="invalid_context")

    # Load KB. If loading fails, no automatic actuator command is permitted.
    try:
        if kb is None:
            if get_knowledge_base is None:
                raise RuntimeError("KnowledgeBase tidak dapat diimpor")
            kb = get_knowledge_base()
        topics = _kb_topics(kb)
        kb_source = "unknown"
        if hasattr(kb, "get_loaded_source"):
            try:
                kb_source = str(kb.get_loaded_source() or "unknown")
            except Exception:
                pass
        irrigation_topic = topics.get("irrigation")
        fertilization_topic = topics.get("fertilization")
        soil_topic = topics.get("soil")
        if not all(isinstance(topic, dict) for topic in (irrigation_topic, fertilization_topic, soil_topic)):
            raise ValueError("Topik KB wajib irrigation, fertilization, dan soil tidak lengkap")
        moisture_min, moisture_max = _extract_moisture_limits(irrigation_topic)
        ph_min, ph_max = _extract_ph_tolerance(soil_topic)
        if moisture_min is None or moisture_max is None:
            raise ValueError("Ambang irrigation.soil_moisture.ideal.min/max tidak valid")
        if ph_min is None or ph_max is None:
            raise ValueError("Rentang toleransi pH tidak ditemukan atau tidak valid di KB soil")
    except Exception as exc:
        return _result(
            reason=f"Knowledge Base tidak valid/tersedia; keputusan otomatis dihentikan: {exc}",
            analysis="Rule Engine tidak memakai ambang hardcoded saat data KB inti tidak valid.",
            risk="Konfigurasi agronomi tidak tervalidasi.",
            recommendation="Periksa struktur /knowledge_base/irrigation, /knowledge_base/fertilization, dan /knowledge_base/soil di Firebase.",
            blocked_reason="knowledge_base_invalid",
        )

    # Read sensor values and determine agronomic need using the KB's values.
    moisture = _number(context.get("soil_moisture"))
    if moisture is None or not (0 <= moisture <= 100):
        return _result(
            reason="Kelembapan tanah tidak tersedia atau di luar rentang 0–100%; aktuator ditahan.",
            analysis=f"Ambang kelembapan dari KB: {moisture_min:g}–{moisture_max:g}%.",
            risk="Sensor kelembapan tidak valid.",
            recommendation="Periksa sensor dan nilai /sensor/soil di Firebase.",
            kb_source=kb_source,
            blocked_reason="invalid_soil_moisture",
        )

    irrigation_needed = moisture < moisture_min
    # Humidity above the ideal max is explicitly NOT a signal to irrigate or fertilize.
    moisture_explanation = (
        f"Kelembapan aktual {moisture:.1f}% di bawah batas minimum KB {moisture_min:g}%"
        if irrigation_needed else
        f"Kelembapan aktual {moisture:.1f}% berada di atas batas maksimum KB {moisture_max:g}%"
        if moisture > moisture_max else
        f"Kelembapan aktual {moisture:.1f}% berada dalam rentang ideal KB {moisture_min:g}–{moisture_max:g}%"
    )

    # Fertilization is only evaluated when a supported crop/field profile is set.
    profile = str(context.get("farming_profile") or os.getenv("FARMING_PROFILE", "")).strip()
    start_age, interval_days, schedule_message = _extract_npk_schedule(fertilization_topic, profile)
    age_days = _plant_age(context)
    days_since_last = _fertilization_days_since(context)
    fertilization_needed = False
    fertilization_unavailable_reason = ""
    if start_age is None or interval_days is None:
        fertilization_unavailable_reason = schedule_message
    elif age_days is None:
        fertilization_unavailable_reason = "Umur tanaman (HST) tidak tersedia; jadwal pemupukan tidak dapat dihitung."
    elif age_days < start_age:
        fertilization_unavailable_reason = f"Umur tanaman {age_days} HST belum mencapai jadwal pemupukan mulai {start_age} HST."
    elif days_since_last is None or days_since_last < 0:
        fertilization_unavailable_reason = "Riwayat pemupukan terakhir tidak tersedia/invalid; pemupukan ditahan."
    elif days_since_last < interval_days:
        fertilization_unavailable_reason = f"Pemupukan belum jatuh tempo: baru {days_since_last:g} hari sejak aplikasi terakhir; interval minimum KB {interval_days} hari."
    else:
        fertilization_needed = True
        fertilization_unavailable_reason = f"Pemupukan jatuh tempo: umur {age_days} HST dan {days_since_last:g} hari sejak aplikasi terakhir."

    # Gather all actuation safety gates. Unknown values are not treated as safe.
    now_value = time.time() if now is None else now
    yolo_valid, yolo_message = _yolo_status(context.get("plant_detection"), now_value)
    local_rain = _bool_value(context.get("rain"))
    rain_probability = _rain_probability(context)

    gate_reasons = []
    if not yolo_valid:
        gate_reasons.append(yolo_message)
    if local_rain is None:
        gate_reasons.append("Status sensor hujan lokal tidak tersedia/invalid.")
    elif local_rain:
        gate_reasons.append("Sensor hujan lokal mendeteksi hujan.")
    if rain_probability is None:
        gate_reasons.append("Probabilitas hujan OpenWeatherMap tidak tersedia/invalid.")
    elif rain_probability >= OWM_RAIN_PROBABILITY_THRESHOLD:
        gate_reasons.append(
            f"Probabilitas hujan OpenWeatherMap {rain_probability:.1f}% mencapai ambang {OWM_RAIN_PROBABILITY_THRESHOLD:.0f}%.")

    # pH is a fertilization gate only; it does not change irrigation need.
    ph = _number(context.get("ph"))
    ph_valid = ph is not None and ph_min <= ph <= ph_max

    intended_action = "NONE"
    if irrigation_needed:
        intended_action = "IRRIGATE"
    elif fertilization_needed:
        intended_action = "FERTILIZE"

    action = intended_action
    blocked_reason = ""
    if action in {"IRRIGATE", "FERTILIZE"} and gate_reasons:
        action = "NONE"
        blocked_reason = "safety_gate: " + " ".join(gate_reasons)

    if action == "FERTILIZE" and not ph_valid:
        action = "NONE"
        if ph is None:
            blocked_reason = "pH sensor tidak tersedia/invalid; pemupukan ditahan."
        else:
            blocked_reason = f"pH {ph:.2f} di luar rentang toleransi KB {ph_min:g}–{ph_max:g}; pemupukan ditahan."

    if action == "IRRIGATE":
        reason = f"IRRIGATE: {moisture_explanation}; YOLO dan seluruh gate hujan valid."
        recommendation = "Lakukan penyiraman sesuai durasi yang dibatasi di ai_orchestrator/ESP32."
        safety_status = "READY"
    elif action == "FERTILIZE":
        reason = f"FERTILIZE: {fertilization_unavailable_reason} pH {ph:.2f} berada dalam rentang KB {ph_min:g}–{ph_max:g}; YOLO dan seluruh gate hujan valid."
        recommendation = "Jalankan satu siklus servo sesuai kalibrasi dosis mekanis yang sudah diuji."
        safety_status = "READY"
    elif intended_action != "NONE":
        reason = f"Kebutuhan agronomis terdeteksi tetapi aktuator ditahan. {blocked_reason}"
        recommendation = "Pulihkan input safety yang bermasalah; perintah tidak dikirim ke aktuator."
        safety_status = "BLOCKED"
    elif not fertilization_needed and fertilization_unavailable_reason:
        reason = f"{moisture_explanation}. Tidak ada penyiraman yang dibutuhkan. {fertilization_unavailable_reason}"
        recommendation = "Pantau sensor dan riwayat pemupukan; tidak ada perintah aktuator saat ini."
        safety_status = "READY" if not gate_reasons else "BLOCKED"
        if gate_reasons:
            blocked_reason = "Safety input bermasalah: " + " ".join(gate_reasons)
    else:
        reason = f"{moisture_explanation}. Tidak ada tindakan pemupukan yang jatuh tempo."
        recommendation = "Lanjutkan pemantauan berkala."
        safety_status = "READY" if not gate_reasons else "BLOCKED"
        if gate_reasons:
            blocked_reason = "Safety input bermasalah: " + " ".join(gate_reasons)

    analysis_parts = [moisture_explanation + ".", schedule_message]
    if age_days is not None:
        analysis_parts.append(f"Umur tanaman: {age_days} HST.")
    if days_since_last is not None:
        analysis_parts.append(f"Hari sejak pemupukan terakhir: {days_since_last:g}.")
    if ph is not None:
        analysis_parts.append(f"pH aktual: {ph:.2f}; rentang toleransi KB: {ph_min:g}–{ph_max:g}.")
    if gate_reasons:
        analysis_parts.append("Safety gate: " + " ".join(gate_reasons))

    risk_parts = []
    if moisture > moisture_max:
        risk_parts.append("Kelembapan tanah melebihi rentang ideal; jangan menambah penyiraman hanya karena nilai ini tinggi.")
    if ph is not None and not ph_valid:
        risk_parts.append("pH di luar rentang toleransi KB; pemupukan diblokir, tetapi keputusan penyiraman tidak diubah.")
    if fertilization_unavailable_reason and not fertilization_needed:
        risk_parts.append(fertilization_unavailable_reason)
    if gate_reasons:
        risk_parts.extend(gate_reasons)

    return _result(
        action=action,
        irrigation_needed=irrigation_needed,
        fertilization_needed=fertilization_needed,
        reason=reason,
        analysis=" ".join(analysis_parts),
        risk=" ".join(risk_parts) if risk_parts else "Tidak ada risiko tambahan yang terdeteksi dari aturan yang dievaluasi.",
        recommendation=recommendation,
        kb_source=kb_source,
        safety_status=safety_status,
        blocked_reason=blocked_reason,
    )


# Alias for callers that prefer a verb-style name.
run_rule_engine = evaluate_rules
