
import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = None

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    client = Client(
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN
    )


def send_message(text):
    """
    Menampilkan pesan ke terminal.
    Fungsi ini tidak mengirim pesan ke WhatsApp.
    """
    print("WHATSAPP MESSAGE:")
    print(text)


def _send_whatsapp(to, message=None, content_sid=None, content_variables=None):
    """
    Mengirim satu pesan melalui Twilio.
    Bisa mengirim pesan teks biasa (body) atau template (content_sid).
    """
    if client is None or not FROM:
        print("WHATSAPP ERROR: Konfigurasi Twilio belum lengkap.")
        return False

    try:
        # 🟢 PERBAIKAN: Pastikan format nomor 'to' bersih dan tidak double 'whatsapp:'
        to_str = str(to).strip()
        if to_str.startswith("whatsapp:"):
            # Jika sudah ada prefix whatsapp:, ambil nomornya saja
            clean_to = to_str 
        else:
            # Jika belum ada, tambahkan prefix whatsapp:
            clean_to = f"whatsapp:{to_str}"

        # 🟢 PERBAIKAN: Pastikan format nomor 'FROM' di .env juga benar
        from_str = str(FROM).strip()
        clean_from = from_str if from_str.startswith("whatsapp:") else f"whatsapp:{from_str}"

        # Menyiapkan argumen dasar
        payload = {
            "from_": clean_from,
            "to": clean_to
        }

        # JIKA MENGGUNAKAN TEMPLATE
        if content_sid:
            payload["content_sid"] = content_sid
            if content_variables:
                payload["content_variables"] = content_variables
        
        # JIKA MENGGUNAKAN TEKS BEBAS (Hasil Gemini)
        elif message:
            payload["body"] = message
        
        else:
            print("WHATSAPP ERROR: Harus menyertakan message atau content_sid.")
            return False

        # Kirim request ke Twilio
        msg = client.messages.create(**payload)

        print("WhatsApp sent:", msg.sid)
        return True

    except TwilioRestException as e:
        print(
            f"TWILIO ERROR: code={e.code}, "
            f"status={e.status}, message={e.msg}"
        )
        if e.code == 21654:
            print("Tips: Periksa apakah session 24 jam untuk nomor ini benar-benar aktif.")
        return False
    except Exception as e:
        print(f"WHATSAPP ERROR: {type(e).__name__}: {e}")
        return False

def send_whatsapp(to, message):
    """Mengirim satu pesan WhatsApp."""
    return _send_whatsapp(to, message)


def send_whatsapp_chunked(to, message, chunk_size=1500):
    """
    Memecah pesan panjang menjadi beberapa bagian.
    Pengiriman dihentikan jika salah satu bagian gagal.
    """
    if not isinstance(message, str) or not message:
        print("WHATSAPP ERROR: Pesan kosong atau tidak valid.")
        return False

    if chunk_size <= 0:
        print("WHATSAPP ERROR: chunk_size harus lebih dari 0.")
        return False

    chunks = [
        message[i:i + chunk_size]
        for i in range(0, len(message), chunk_size)
    ]

    for i, chunk in enumerate(chunks, start=1):
        prefix = f"({i}/{len(chunks)})\n"

        success = _send_whatsapp(
            to=to,
            message=prefix + chunk
        )

        if not success:
            print(
                f"Pengiriman dihentikan pada bagian "
                f"{i}/{len(chunks)}."
            )
            return False

    return True
