import requests
import time

from config import (
    OPENWEATHER_API_KEY,
    LAT,
    LON
)

from firebase_service import update_data, get_data


def update_weather():

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={LAT}"
        f"&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )

    try:

        response = requests.get(url)

        if response.status_code != 200:
            print("Weather API Error:",
                  response.status_code)
            return None

        data = response.json()

        # Ambil prediksi 3 jam ke depan
        forecast = data["list"][0]

        rain_probability = (
            forecast["pop"] * 100
        )

        weather_data = {

            "rain_probability":
                round(rain_probability, 1),

            "forecast":
                forecast["weather"][0]["main"],

            "temperature":
                forecast["main"]["temp"],

            "humidity":
                forecast["main"]["humidity"],

            "rain_3h":
                forecast.get(
                    "rain", {}
                ).get("3h", 0),

            "timestamp":
                int(time.time())
        }

        update_data(
            "weather",
            weather_data
        )

        print(
            "Weather updated:",
            weather_data
        )

        return weather_data

    except Exception as e:

        print(
            "Weather Error:",
            e
        )

        return None

def get_weather_forecast():
    """
    Membaca prakiraan cuaca yang tersimpan
    di Firebase agar semua modul menggunakan
    sumber data yang sama.
    """
    weather = get_data("weather")

    if not isinstance(weather, dict):
        return None

    if weather.get("rain_probability") is None:
        return None

    return {
        "rain_probability": weather["rain_probability"],
        "temperature": weather.get("temperature"),
        "humidity": weather.get("humidity"),
        "condition": weather.get(
            "forecast",
            "UNKNOWN"
        ),
        "rain_3h": weather.get("rain_3h"),
        "timestamp": weather.get("timestamp"),
    }
