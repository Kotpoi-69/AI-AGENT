
"""orchestrator_api.py

Endpoint pemicu satu siklus Rule Engine untuk cron-job.org.
Tidak menjalankan loop otomatis.
"""

import asyncio
import hmac
import os
from threading import Lock

from fastapi import FastAPI, Header, HTTPException

from ai_orchestrator import run_ai_cycle

app = FastAPI(
    title="AIoT Knowledge-Based Rule Engine",
    version="1.0.0",
)

# Mencegah dua siklus berjalan bersamaan dalam satu proses.
cycle_lock = Lock()


@app.get("/")
def root():
    return {
        "service": "AIoT Knowledge-Based Rule Engine",
        "status": "online",
        "trigger": "POST /run-cycle",
    }


@app.get("/health")
def health():
    # Hanya pemeriksaan kesehatan; tidak menjalankan Rule Engine.
    return {"status": "ok"}


@app.post("/run-cycle")
async def trigger_cycle(
    x_cron_secret: str = Header(default="", alias="X-Cron-Secret"),
):
    expected_secret = os.getenv("CRON_SECRET", "")

    if not expected_secret:
        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET belum dikonfigurasi.",
        )

    if not hmac.compare_digest(x_cron_secret, expected_secret):
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid.",
        )

    # Tolak pemicu baru jika siklus sebelumnya belum selesai.
    if not cycle_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Siklus sebelumnya masih berjalan.",
        )

    try:
        # Menjalankan fungsi sinkron tanpa memblokir event loop FastAPI.
        result = await asyncio.to_thread(run_ai_cycle)

        decision = result.get("ai_result", {})
        return {
            "ok": True,
            "cycle_completed": True,
            "action": decision.get("action", "NONE"),
            "safety_status": decision.get("safety_status"),
            "blocked_reason": decision.get("blocked_reason"),
            "published": result.get("published", False),
            "publish_message": result.get("publish_message", ""),
        }

    except Exception:
        # Detail exception tetap tersedia di log server.
        import logging
        logging.exception("Siklus Rule Engine gagal.")

        raise HTTPException(
            status_code=500,
            detail="Siklus Rule Engine gagal. Periksa log Render.",
        )

    finally:
        cycle_lock.release()
