# agri_ai_agent.py

from firebase_service import get_data

from weather_service import (
    get_weather_forecast
)

from gemini_service import (
    chat_with_ai
)

# ==========================================
# LOAD CONTEXT
# ==========================================

def get_context():
    return {
        "sensor": get_data("sensor") or {},
        "weather": get_weather_forecast(),
        "agent": get_data("agent") or {},
        "cam": get_data("cam") or {},
    }


# ==========================================
# AGRICULTURE CHAT AGENT
# ==========================================

def agriculture_chat_agent(user_message: str):
    context = get_context()

    sensor = context["sensor"]
    weather = context["weather"]
    agent = context["agent"]
    cam = context["cam"]

    # ======================================
    # PROMPT
    # ======================================

    prompt = f"""
Anda adalah AI Agriculture Consultant profesional.

Gunakan HANYA data berikut.

==================================
DATA SENSOR
==================================

Soil Moisture :
{sensor.get('soil', '-')}

Temperature :
{sensor.get('temperature', '-')}

Humidity :
{sensor.get('humidity', '-')}

pH :
{sensor.get('ph', '-')}

Rain Sensor :
{sensor.get('rain', '-')}

==================================
DATA CUACA
==================================

Rain Probability :
{weather.get('rain_probability', '-')}

Temperature :
{weather.get('temperature', '-')}

Humidity :
{weather.get('humidity', '-')}

Condition :
{weather.get('condition', '-')}

==================================
DATA AI AGENT
==================================

Status :
{agent.get('status', '-')}

Recommendation :
{agent.get('recommendation', '-')}

Reason :
{agent.get('reason', '-')}

Pump Duration :
{agent.get('pump_duration', '-')}

Plant Age :
{agent.get('age', '-')}

==================================
DATA TANAMAN
==================================

Detected :
{cam.get('tanaman', '-')}

Confidence :
{cam.get('confidence', '-')}

Jumlah Tanaman :
{agent.get('plant_count', '-')}

Umur Tanaman :
{agent.get('age_days', '-')}

==================================
USER QUESTION
==================================

{user_message}

ATURAN:

1. Jangan mengarang data.
2. Gunakan data aktual di atas.
3. Jika data tidak tersedia, katakan tidak tersedia.
4. Berikan jawaban seperti konsultan pertanian profesional.
5. Jika user bertanya panen, umur tanaman atau jumlah tanaman,
   gunakan DATA PRODUKSI.
6. Jika user bertanya kondisi lahan,
   gunakan sensor + cuaca + AI agent.
"""

    try:
        response = chat_with_ai(prompt)

        if not response or not response.strip():
            return (
                "⚠️ Layanan AI sedang tidak tersedia karena "
                "respons Gemini kosong atau kuota API tercapai. "
                "Silakan coba kembali setelah kuota tersedia."
            )

        return response

    except Exception as e:
        print("AGRI CHAT ERROR:", e)

        return (
            "⚠️ AI sedang tidak tersedia. "
            "Silakan coba beberapa saat lagi."
        )
