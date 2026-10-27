
import os
import json
import re
from datetime import datetime, timezone

from google import genai

from knowledge_base import KnowledgeBase


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ============================================================
# CONFIGURATION
# ============================================================

VALID_ACTIONS = {
    "NONE",
    "IRRIGATE",
    "FERTILIZE",
}

# ------------------------------------------------------------
# YOLO SAFETY
# ------------------------------------------------------------

MIN_YOLO_CONFIDENCE = 0.25
MAX_YOLO_AGE_SECONDS = 300

# ------------------------------------------------------------
# WEATHER SAFETY
# ------------------------------------------------------------

# Jika probabilitas hujan OWM >= 50%, maka:
# - irigasi ditunda
# - pemupukan ditunda
OWM_RAIN_PROBABILITY_THRESHOLD = 50.0

# ------------------------------------------------------------
# FERTILIZATION SAFETY
# ------------------------------------------------------------

# Ini adalah technical cooldown untuk melindungi aktuator.
# BUKAN interval agronomis dari Knowledge Base.
FERTILIZATION_INTERVAL_DAYS = 7

# Technical pH gate untuk aktuasi pupuk.
FERTILIZATION_PH_MIN = 5.5
FERTILIZATION_PH_MAX = 6.3


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

kb = KnowledgeBase()


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_text(value, default=""):
    """
    Mengubah nilai menjadi string secara aman.
    """
    if value is None:
        return default

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _to_bool(value, default=False):
    """
    Konversi nilai menjadi boolean secara aman.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        value = value.strip().lower()

        if value in {
            "true",
            "1",
            "yes",
            "ya",
            "on",
            "aktif",
            "detected",
            "terdeteksi",
        }:
            return True

        if value in {
            "false",
            "0",
            "no",
            "tidak",
            "off",
            "nonaktif",
            "not detected",
            "tidak terdeteksi",
        }:
            return False

    return default


def _to_float(value, default=0.0):
    """
    Konversi nilai menjadi float secara aman.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_json(text):
    """
    Mengambil JSON dari respons Gemini.

    Mendukung:
    1. JSON langsung
    2. JSON dalam markdown code block
    3. JSON yang dikelilingi teks tambahan
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Markdown code block
    # --------------------------------------------------------

    code_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # Find first JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# YOLO
# ============================================================

def _get_plant_detection(context):
    """
    Mengambil data deteksi tanaman dari context.
    """

    detection = context.get(
        "plant_detection",
        {}
    )

    if not isinstance(detection, dict):
        detection = {}

    return detection


def _get_yolo_status(context):
    """
    Menentukan apakah hasil YOLO masih valid untuk safety gate.

    Valid jika:
    - plant_detected == True
    - confidence >= MIN_YOLO_CONFIDENCE
    - timestamp tersedia
    - umur data <= MAX_YOLO_AGE_SECONDS
    """

    detection = _get_plant_detection(context)

    plant_detected = _to_bool(
        detection.get("plant_detected"),
        default=False
    )

    confidence = _to_float(
        detection.get("confidence"),
        default=0.0
    )

    timestamp = detection.get("timestamp")

    # --------------------------------------------------------
    # Tidak ada timestamp
    # --------------------------------------------------------

    if timestamp is None:

        return {
            "valid": False,
            "plant_detected": plant_detected,
            "confidence": confidence,
            "age_seconds": None,
            "reason": (
                "Data YOLO tidak memiliki timestamp."
            ),
        }

    # --------------------------------------------------------
    # Hitung umur data
    # --------------------------------------------------------

    try:
        timestamp = float(timestamp)

        now = datetime.now(
            timezone.utc
        ).timestamp()

        age_seconds = max(
            0,
            now - timestamp
        )

    except (TypeError, ValueError):

        return {
            "valid": False,
            "plant_detected": plant_detected,
            "confidence": confidence,
            "age_seconds": None,
            "reason": (
                "Timestamp YOLO tidak valid."
            ),
        }

    # --------------------------------------------------------
    # Tanaman tidak terdeteksi
    # --------------------------------------------------------

    if not plant_detected:

        return {
            "valid": False,
            "plant_detected": False,
            "confidence": confidence,
            "age_seconds": age_seconds,
            "reason": (
                "Tanaman tidak terdeteksi oleh YOLO."
            ),
        }

    # --------------------------------------------------------
    # Confidence terlalu rendah
    # --------------------------------------------------------

    if confidence < MIN_YOLO_CONFIDENCE:

        return {
            "valid": False,
            "plant_detected": True,
            "confidence": confidence,
            "age_seconds": age_seconds,
            "reason": (
                f"Confidence YOLO terlalu rendah "
                f"({confidence:.2f} < "
                f"{MIN_YOLO_CONFIDENCE:.2f})."
            ),
        }

    # --------------------------------------------------------
    # Data terlalu lama
    # --------------------------------------------------------

    if age_seconds > MAX_YOLO_AGE_SECONDS:

        return {
            "valid": False,
            "plant_detected": True,
            "confidence": confidence,
            "age_seconds": age_seconds,
            "reason": (
                f"Data YOLO sudah terlalu lama "
                f"({age_seconds:.1f} detik)."
            ),
        }

    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    return {
        "valid": True,
        "plant_detected": True,
        "confidence": confidence,
        "age_seconds": age_seconds,
        "reason": (
            "Tanaman terdeteksi dan data YOLO "
            "masih valid."
        ),
    }


def _plant_detected(context):
    """
    Compatibility helper.
    """

    return _get_yolo_status(context)["valid"]


# ============================================================
# RAIN / WEATHER SAFETY
# ============================================================

def _rain_detected(context):
    """
    Sensor hujan lokal adalah kondisi hujan aktual.

    Sensor hujan memblokir:
    - IRRIGATE
    - FERTILIZE
    """

    sensor = context.get(
        "sensor",
        {}
    )

    if not isinstance(sensor, dict):
        return False

    return _to_bool(
        sensor.get("rain"),
        default=False
    )


def _get_rain_probability(context):
    """
    Mengambil probabilitas hujan dari OpenWeatherMap.

    Nilai default = 0 agar sistem tidak melakukan
    pemblokiran jika data OWM tidak tersedia.
    """

    weather = context.get(
        "weather",
        {}
    )

    if not isinstance(weather, dict):
        return 0.0

    return _to_float(
        weather.get("rain_probability"),
        default=0.0
    )


def _high_rain_probability(context):
    """
    Menentukan apakah probabilitas hujan OWM cukup tinggi
    untuk menunda irigasi dan pemupukan.

    Threshold sistem:
        >= 50%
    """

    rain_probability = _get_rain_probability(
        context
    )

    return (
        rain_probability
        >= OWM_RAIN_PROBABILITY_THRESHOLD
    )


def _get_rain_safety_status(context):
    """
    Menggabungkan kondisi hujan aktual dan prediksi hujan.

    rain_block = True jika:
    - sensor hujan lokal aktif, ATAU
    - probabilitas hujan OWM >= 50%
    """

    local_rain = _rain_detected(
        context
    )

    rain_probability = _get_rain_probability(
        context
    )

    high_rain_probability = (
        rain_probability
        >= OWM_RAIN_PROBABILITY_THRESHOLD
    )

    blocked = (
        local_rain
        or high_rain_probability
    )

    reasons = []

    if local_rain:
        reasons.append(
            "sensor hujan lokal aktif"
        )

    if high_rain_probability:
        reasons.append(
            f"probabilitas hujan OWM "
            f"{rain_probability:.1f}% "
            f"(threshold "
            f"{OWM_RAIN_PROBABILITY_THRESHOLD:.0f}%)"
        )

    if blocked:
        reason = (
            " dan ".join(reasons)
            + "."
        )
    else:
        reason = (
            f"tidak ada hujan lokal dan "
            f"probabilitas hujan OWM "
            f"{rain_probability:.1f}% berada "
            f"di bawah threshold."
        )

    return {
        "blocked": blocked,
        "local_rain": local_rain,
        "rain_probability": rain_probability,
        "high_rain_probability": high_rain_probability,
        "reason": reason,
    }


# ============================================================
# pH
# ============================================================

def _get_ph(context):
    """
    Mengambil pH tanah dari sensor.
    """

    sensor = context.get(
        "sensor",
        {}
    )

    if not isinstance(sensor, dict):
        return None

    value = sensor.get("ph")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ph_eligible_for_fertilization(context):
    """
    Technical gate pH untuk pemupukan.

    Catatan:
    Fungsi ini TIDAK menentukan apakah tanaman
    membutuhkan pupuk.

    Fungsi ini hanya menentukan apakah aktuator pupuk
    secara teknis boleh dijalankan.
    """

    ph = _get_ph(context)

    if ph is None:

        return {
            "eligible": False,
            "ph": None,
            "reason": (
                "Nilai pH tidak tersedia."
            ),
        }

    if (
        ph < FERTILIZATION_PH_MIN
        or ph > FERTILIZATION_PH_MAX
    ):

        return {
            "eligible": False,
            "ph": ph,
            "reason": (
                f"pH {ph:.2f} berada di luar "
                f"rentang technical gate "
                f"{FERTILIZATION_PH_MIN:.1f}–"
                f"{FERTILIZATION_PH_MAX:.1f}."
            ),
        }

    return {
        "eligible": True,
        "ph": ph,
        "reason": (
            f"pH {ph:.2f} memenuhi "
            "technical gate pemupukan."
        ),
    }


# ============================================================
# FERTILIZATION CONTEXT
# ============================================================

def _get_fertilization_context(context):
    """
    Mengambil informasi pemupukan jika tersedia.

    Data ini dapat berisi:
    - allowed
    - due
    - interval_days
    - days_since_last

    Interval agronomis tetap berasal dari KB/agent.
    """

    fertilization = context.get(
        "fertilization",
        {}
    )

    if not isinstance(fertilization, dict):
        fertilization = {}

    return fertilization


def _fertilization_allowed(context):
    """
    Technical permission tambahan jika tersedia.

    Jika field 'allowed' tidak ada,
    default = True karena keputusan kebutuhan
    berasal dari agent.

    Catatan:
    Ini bukan keputusan agronomis.
    """

    fertilization = _get_fertilization_context(
        context
    )

    if "allowed" not in fertilization:
        return True

    return _to_bool(
        fertilization.get("allowed"),
        default=True
    )


def _fertilization_interval_due(context):
    """
    Mengecek technical cooldown jika informasi
    riwayat pemupukan memang tersedia.

    Fungsi ini TIDAK menentukan kebutuhan pupuk
    secara agronomis.

    Jika days_since_last tidak tersedia,
    jangan mengarang status due.
    """

    fertilization = _get_fertilization_context(
        context
    )

    days_since_last = fertilization.get(
        "days_since_last"
    )

    if days_since_last is None:

        return {
            "known": False,
            "due": None,
            "reason": (
                "Riwayat pemupukan terakhir "
                "tidak tersedia."
            ),
        }

    try:
        days_since_last = float(
            days_since_last
        )

    except (TypeError, ValueError):

        return {
            "known": False,
            "due": None,
            "reason": (
                "Data riwayat pemupukan "
                "tidak valid."
            ),
        }

    due = (
        days_since_last
        >= FERTILIZATION_INTERVAL_DAYS
    )

    if due:

        reason = (
            f"Sudah {days_since_last:.1f} hari "
            "sejak pemupukan terakhir."
        )

    else:

        reason = (
            f"Baru {days_since_last:.1f} hari "
            "sejak pemupukan terakhir."
        )

    return {
        "known": True,
        "due": due,
        "days_since_last": days_since_last,
        "reason": reason,
    }


def _get_fertilization_gate(context):
    """
    Technical safety gate untuk aktuator pemupukan.

    Gate:
    1. YOLO valid
    2. pH memenuhi technical range
    3. optional 'allowed' tidak false
    4. optional cooldown jika riwayat tersedia

    Gate ini TIDAK menentukan:
        fertilization_needed

    Kebutuhan pupuk tetap ditentukan oleh:
        Gemini + Knowledge Base
    """

    yolo_status = _get_yolo_status(
        context
    )

    if not yolo_status["valid"]:

        return {
            "eligible": False,
            "reason": (
                f"YOLO tidak valid: "
                f"{yolo_status['reason']}"
            ),
        }

    ph_status = _ph_eligible_for_fertilization(
        context
    )

    if not ph_status["eligible"]:

        return {
            "eligible": False,
            "reason": ph_status["reason"],
        }

    if not _fertilization_allowed(
        context
    ):

        return {
            "eligible": False,
            "reason": (
                "Pemupukan diblokir oleh "
                "technical permission."
            ),
        }

    
    interval_status = (
        _fertilization_interval_due(
            context
        )
    )

    # BLOKIR jika riwayat tidak tersedia atau tidak valid.
    if not interval_status["known"]:
        return {
            "eligible": False,
            "reason": (
                "Pemupukan diblokir karena "
                "riwayat pemupukan terakhir "
                "tidak tersedia atau tidak valid. "
                "Periksa data Firebase "
                "/fertilization_history/latest."
            ),
        }

    # BLOKIR jika cooldown 7 hari belum terpenuhi.
    if not interval_status["due"]:
        return {
            "eligible": False,
            "reason": (
                "Technical cooldown pemupukan "
                "belum terpenuhi. "
                f"{interval_status['reason']}"
            ),
        }


    return {
        "eligible": True,
        "reason": (
            "Technical safety gate pemupukan "
            "terpenuhi."
        ),
    }


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def build_knowledge_context(
    purpose="general",
    force_refresh=False,
):
    """
    Mengambil context Knowledge Base.
    """

    try:

        return kb.build_context(
            purpose=purpose,
            force_refresh=force_refresh,
        )

    except Exception as error:

        return {
            "purpose": purpose,
            "source": "Knowledge Base",
            "loaded_from": "error",
            "knowledge": {},
            "error": str(error),
        }


# ============================================================
# GEMINI PROMPT
# ============================================================

def _build_agent_prompt(
    context,
    purpose="analysis",
):
    """
    Prompt utama untuk Knowledge-Based AI Agent.

    Prinsip:
    - KB = sumber utama aturan agronomis.
    - Gemini = menentukan kebutuhan agronomis.
    - Python = technical safety validator.
    """

    kb_context = build_knowledge_context(
        purpose=purpose
    )

    return f"""
Anda adalah Knowledge-Based AI Agent untuk sistem
pertanian cerdas budidaya tanaman cabai.

Tugas utama Anda adalah melakukan ANALISIS AGRONOMIS
berdasarkan Knowledge Base dan kondisi aktual tanaman.

============================================================
ATURAN UTAMA
============================================================

1. Knowledge Base adalah sumber utama untuk aturan
   budidaya cabai.

2. Tentukan sendiri berdasarkan Knowledge Base apakah:
   - tanaman membutuhkan irigasi
   - tanaman membutuhkan pemupukan

3. Field:
   - irrigation_needed
   - fertilization_needed

   harus merepresentasikan KEBUTUHAN AGRONOMIS,
   bukan status apakah aktuator saat ini aman dijalankan.

4. Jangan mengubah kebutuhan agronomis menjadi false
   hanya karena safety gate memblokir aktuasi.

5. Python safety layer akan melakukan validasi teknis
   setelah respons Anda.

6. Jika terdapat aturan jadwal budidaya berdasarkan HST
   yang secara eksplisit terdapat pada Knowledge Base,
   aturan tersebut boleh digunakan.

7. Jika Knowledge Base menyebut interval pemupukan
   berdasarkan jarak hari sejak pemupukan terakhir,
   jangan menyatakan interval tersebut sudah jatuh tempo
   jika tanggal/riwayat pemupukan terakhir tidak tersedia.

8. Jangan mengarang data yang tidak tersedia.

9. pH rendah tidak otomatis berarti:
      fertilization_needed = true

   karena koreksi pH dapat memerlukan tindakan seperti
   pengapuran dan tidak otomatis berarti pemberian NPK.

10. OWM digunakan sebagai informasi prakiraan cuaca.
    Namun keputusan kebutuhan agronomis tetap berdasarkan
    Knowledge Base dan kondisi tanaman.

11. Probabilitas hujan OWM >= 50% merupakan TECHNICAL
    SAFETY GATE yang akan diproses Python untuk menunda
    irigasi dan pemupukan.

12. Sensor hujan lokal aktif juga merupakan TECHNICAL
    SAFETY GATE yang akan diproses Python untuk menunda
    irigasi dan pemupukan.

13. YOLO plant detection merupakan safety gate.
    Jika tanaman tidak terdeteksi, confidence rendah,
    atau data terlalu lama, aktuasi akan diblokir oleh
    Python.

14. Jangan memalsukan status safety gate.

============================================================
INPUT SISTEM
============================================================

DATA SENSOR:
{json.dumps(
    context.get("sensor", {}),
    indent=2,
    ensure_ascii=False
)}

DATA CUACA OWM:
{json.dumps(
    context.get("weather", {}),
    indent=2,
    ensure_ascii=False
)}

DATA TANAMAN:
{json.dumps(
    context.get("plant", {}),
    indent=2,
    ensure_ascii=False
)}

DATA DETEKSI YOLO:
{json.dumps(
    context.get("plant_detection", {}),
    indent=2,
    ensure_ascii=False
)}

DATA PEMUPUKAN:
{json.dumps(
    context.get("fertilization", {}),
    indent=2,
    ensure_ascii=False
)}

============================================================
KNOWLEDGE BASE
============================================================

{json.dumps(
    kb_context,
    indent=2,
    ensure_ascii=False
)}

============================================================
OUTPUT
============================================================

Kembalikan HANYA JSON valid dengan struktur:

{{
  "action": "NONE | IRRIGATE | FERTILIZE",
  "irrigation_needed": true,
  "fertilization_needed": false,
  "reason": "alasan analisis agronomis"
}}

PENTING:

- action adalah rekomendasi awal berdasarkan analisis.
- Python akan melakukan safety validation.
- Jangan menghapus irrigation_needed hanya karena hujan.
- Jangan menghapus fertilization_needed hanya karena hujan.
- Jangan menganggap OWM rain_probability sebagai kondisi
  hujan aktual.
- Jangan mengaktifkan aktuator hanya karena OWM.
- Jangan membuat keputusan berdasarkan asumsi yang tidak
  terdapat dalam data atau Knowledge Base.
"""


# ============================================================
# GEMINI REQUEST
# ============================================================

def generate_response(prompt, temperature=0.1):
    """
    Mengirim prompt ke Gemini dan mencatat penyebab
    kegagalan tanpa menyamarkan error aslinya.
    """
    try:
        print(f"[GEMINI] Model aktif: {MODEL_NAME}")
        print(f"[GEMINI] Panjang prompt: {len(prompt)} karakter")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": temperature,
            },
        )

        response_text = getattr(response, "text", None)

        # Diagnostik jika SDK tidak menyediakan teks.
        if not response_text or not response_text.strip():
            print("[GEMINI] Respons teks kosong.")
            print(
                "[GEMINI] Candidate info:",
                repr(getattr(response, "candidates", None))
            )
            print(
                "[GEMINI] Prompt feedback:",
                repr(getattr(response, "prompt_feedback", None))
            )
            return ""

        print(
            f"[GEMINI] Respons diterima: "
            f"{len(response_text)} karakter"
        )

        return response_text.strip()

    except Exception as error:
        print(f"[GEMINI] Error type: {type(error).__name__}")
        print(f"[GEMINI] Error detail: {error}")
        return ""

# ============================================================
# DEFAULT ANALYSIS
# ============================================================

def _default_analysis(reason):
    """
    Fallback jika Gemini gagal memberikan JSON.
    """

    return {
        "action": "NONE",
        "irrigation_needed": False,
        "fertilization_needed": False,
        "reason": reason,
    }


# ============================================================
# NORMALIZE DECISION
# ============================================================

def _normalize_decision(result):
    """
    Normalisasi hasil Gemini.
    """

    if not isinstance(result, dict):

        return _default_analysis(
            "Respons agent tidak valid."
        )

    action = _safe_text(
        result.get("action"),
        default="NONE"
    ).upper()

    if action not in VALID_ACTIONS:
        action = "NONE"

    irrigation_needed = _to_bool(
        result.get(
            "irrigation_needed"
        ),
        default=False
    )

    fertilization_needed = _to_bool(
        result.get(
            "fertilization_needed"
        ),
        default=False
    )

    reason = _safe_text(
        result.get("reason"),
        default="Tidak ada alasan."
    )

    return {
        "action": action,
        "irrigation_needed": irrigation_needed,
        "fertilization_needed": fertilization_needed,
        "reason": reason,
    }


# ============================================================
# SAFETY VALIDATION
# ============================================================

def _validate_decision(
    result,
    context,
):
    """
    Technical safety validation.

    Penting:
    Fungsi ini TIDAK menentukan kebutuhan agronomis.

    Fungsi ini hanya menentukan apakah action boleh
    dieksekusi sekarang.

    Safety gate:
    - YOLO
    - sensor hujan
    - OWM rain probability
    - pH
    - technical fertilization cooldown
    """

    result = _normalize_decision(
        result
    )

    action = result["action"]

    irrigation_needed = (
        result["irrigation_needed"]
    )

    fertilization_needed = (
        result["fertilization_needed"]
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo_status = _get_yolo_status(
        context
    )

    if not yolo_status["valid"]:

        result["action"] = "NONE"

        result["reason"] = (
            f"{result['reason']} "
            "Kebutuhan agronomis tetap dicatat "
            "berdasarkan analisis agent, tetapi "
            "seluruh aktuasi diblokir karena "
            f"{yolo_status['reason']}"
        ).strip()

        return result

    # --------------------------------------------------------
    # WEATHER SAFETY
    # --------------------------------------------------------

    rain_status = _get_rain_safety_status(
        context
    )

    rain_block = rain_status["blocked"]

    # --------------------------------------------------------
    # IRRIGATION
    # --------------------------------------------------------

    if action == "IRRIGATE":

        if not irrigation_needed:

            result["action"] = "NONE"

            result["reason"] = (
                "Agent tidak menetapkan kebutuhan "
                "irigasi, sehingga pompa tidak "
                "diaktifkan."
            )

        elif rain_block:

            result["action"] = "NONE"

            result["reason"] = (
                f"{result['reason']} "
                "Kebutuhan irigasi tetap tercatat "
                "berdasarkan analisis agent, tetapi "
                "aktuasi irigasi ditunda karena "
                f"{rain_status['reason']}"
            ).strip()

        else:

            result["action"] = "IRRIGATE"

    # --------------------------------------------------------
    # FERTILIZATION
    # --------------------------------------------------------

    elif action == "FERTILIZE":

        if not fertilization_needed:

            result["action"] = "NONE"

            result["reason"] = (
                "Agent tidak menetapkan kebutuhan "
                "pemupukan, sehingga aktuator "
                "pemupukan tidak diaktifkan."
            )

        elif rain_block:

            result["action"] = "NONE"

            result["reason"] = (
                f"{result['reason']} "
                "Kebutuhan pemupukan tetap tercatat "
                "berdasarkan analisis agent, tetapi "
                "aktuasi pemupukan ditunda karena "
                f"{rain_status['reason']} "
                "untuk mengurangi risiko pupuk hanyut."
            ).strip()

        else:

            fertilization_gate = (
                _get_fertilization_gate(
                    context
                )
            )

            if not fertilization_gate[
                "eligible"
            ]:

                result["action"] = "NONE"

                result["reason"] = (
                    f"{result['reason']} "
                    "Kebutuhan pemupukan tetap "
                    "tercatat berdasarkan analisis "
                    "agent, tetapi aktuasi pemupukan "
                    "diblokir oleh technical safety "
                    "gate: "
                    f"{fertilization_gate['reason']}"
                ).strip()

            else:

                result["action"] = "FERTILIZE"

    # --------------------------------------------------------
    # NONE
    # --------------------------------------------------------

    else:

        result["action"] = "NONE"

    return result


# ============================================================
# FARM ANALYSIS
# ============================================================

def analyze_farm(
    context,
    purpose="analysis",
):
    """
    Analisis utama sistem.

    Alur:

        Context
           ↓
        Knowledge Base
           ↓
        Gemini AI Agent
           ↓
        Agronomic Need
           ↓
        Technical Safety Validation
           ↓
        Final Action
    """

    if not isinstance(context, dict):

        return _default_analysis(
            "Context sistem tidak valid."
        )

    try:

        prompt = _build_agent_prompt(
            context=context,
            purpose=purpose,
        )

        raw_response = generate_response(
            prompt
        )

        if not raw_response:

            return _default_analysis(
                "Gemini tidak memberikan respons."
            )

        parsed = _extract_json(
            raw_response
        )

        if parsed is None:

            return _default_analysis(
                "Respons Gemini tidak dapat "
                "diparsing sebagai JSON."
            )

        normalized = _normalize_decision(
            parsed
        )

        validated = _validate_decision(
            normalized,
            context,
        )

        return validated

    except Exception as error:

        print(
            "ANALYZE FARM ERROR:",
            error
        )

        return _default_analysis(
            f"Terjadi kesalahan analisis: {error}"
        )


# ============================================================
# CHAT WITH AI
# ============================================================

def chat_with_ai(
    message,
    context=None,
):
    """
    Chat umum dengan AI.

    Digunakan untuk pertanyaan WhatsApp
    atau interaksi pengguna lainnya.

    Untuk keputusan aktuator, gunakan analyze_farm().
    """

    if context is None:
        context = {}

    kb_context = build_knowledge_context(
        purpose="general"
    )

    prompt = f"""
Anda adalah Knowledge-Based AI Agent
untuk budidaya tanaman cabai.

Jawab pertanyaan pengguna berdasarkan
Knowledge Base yang tersedia.

Jangan mengarang informasi yang tidak tersedia.
Jawablah pertanyaan pengguna secara padat, ringkas, langsung ke inti masalah, dan MAKSIMAL gunakan 1000 karakter. 
Jangan memberikan penjelasan yang bertele-tele

Jika pertanyaan berkaitan dengan kondisi aktual
tanaman, gunakan data sensor, cuaca, umur tanaman,
dan data YOLO yang diberikan.

Data sistem:
{json.dumps(
    context,
    indent=2,
    ensure_ascii=False
)}

Knowledge Base:
{json.dumps(
    kb_context,
    indent=2,
    ensure_ascii=False
)}

Pertanyaan pengguna:
{message}
"""

    return generate_response(
        prompt
    )


# ============================================================
# SIMPLE ASK AI
# ============================================================

def ask_ai(
    question,
    context=None,
):
    """
    Alias sederhana untuk chat_with_ai().
    """

    return chat_with_ai(
        message=question,
        context=context,
    )



