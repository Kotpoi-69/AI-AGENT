# firebase_service.py
import requests
from config import FIREBASE_URL

TIMEOUT = 15


def _url(path):
    path = str(path).strip("/")
    return f"{FIREBASE_URL.rstrip('/')}/{path}.json"


def get_data(path):
    """Membaca node Firebase RTDB."""
    try:
        response = requests.get(_url(path), timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"GET ERROR {path}: {exc}")
        return None


def update_data(path, data):
    """PATCH node Firebase tanpa menghapus field lain."""
    try:
        response = requests.patch(
            _url(path),
            json=data,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"PATCH ERROR {path}: {exc}")
        return False


def set_data(path, data):
    """PUT: mengganti seluruh isi node Firebase."""
    try:
        response = requests.put(
            _url(path),
            json=data,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"PUT ERROR {path}: {exc}")
        return False
