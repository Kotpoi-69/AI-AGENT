
"""
orchestrator_api.py

API dan background task untuk menjalankan
Knowledge-Based Rule Engine secara berkala.

Background orchestrator berjalan setiap 5 detik.
Endpoint /run-cycle tetap tersedia untuk pemicu manual/eksternal.
"""

import asyncio
import hmac
import os
import logging
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException

from ai_orchestrator import run_ai_cycle


# ============================================================
# KONFIGURASI
# ============================================================

ORCHESTRATOR_INTERVAL = 5

logger = logging.getLogger("orchestrator")

# Mencegah dua siklus berjalan bersamaan.
cycle_lock = asyncio.Lock()


# ============================================================
# BACKGROUND ORCHESTRATOR
# ============================================================

async def orchestrator_loop():

    logger.info(
        "[ORCHESTRATOR] Background loop aktif. Interval: %s detik",
        ORCHESTRATOR_INTERVAL
    )

    while True:

        cycle_start = time.monotonic()

        try:

            # Pastikan tidak ada siklus lain yang sedang berjalan.
            if cycle_lock.locked():

                logger.warning(
                    "[ORCHESTRATOR] Siklus sebelumnya masih berjalan. "
                    "Siklus baru dilewati."
                )

            else:

                async with cycle_lock:

                    logger.info(
                        "[ORCHESTRATOR] Memulai siklus otomatis."
                    )

                    result = await asyncio.to_thread(run_ai_cycle)

                    decision = result.get("ai_result", {})

                    logger.info(
                        "[ORCHESTRATOR] Siklus selesai. "
                        "Action=%s, Published=%s",
                        decision.get("action", "NONE"),
                        result.get("published", False)
                    )

        except Exception:

            logger.exception(
                "[ORCHESTRATOR] Terjadi kesalahan pada background loop."
            )

        # Menjaga interval awal antarsiklus sekitar 5 detik.
        # Jika satu siklus membutuhkan lebih dari 5 detik,
        # siklus berikutnya dimulai setelah siklus sebelumnya selesai.
        elapsed = time.monotonic() - cycle_start

        sleep_time = max(
            0,
            ORCHESTRATOR_INTERVAL - elapsed
        )

        await asyncio.sleep(sleep_time)


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("[API] Memulai AIoT Orchestrator API.")

    background_task = asyncio.create_task(
        orchestrator_loop()
    )

    yield

    logger.info("[API] Menghentikan background orchestrator.")

    background_task.cancel()

    try:
        await background_task

    except asyncio.CancelledError:
        logger.info("[API] Background task berhasil dihentikan.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AIoT Knowledge-Based Rule Engine",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "AIoT Knowledge-Based Rule Engine",
        "status": "online",
        "orchestrator": "running",
        "interval": ORCHESTRATOR_INTERVAL,
        "unit": "seconds"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    # Hanya memeriksa ketersediaan API.
    # Tidak menjalankan Rule Engine.

    return {
        "status": "ok",
        "service": "AIoT Orchestrator"
    }


# ============================================================
# MANUAL / EXTERNAL TRIGGER
# ============================================================

@app.post("/run-cycle")
async def trigger_cycle(
    x_cron_secret: str = Header(
        default="",
        alias="X-Cron-Secret"
    )
):

    expected_secret = os.getenv("CRON_SECRET", "")

    if not expected_secret:

        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET belum dikonfigurasi."
        )

    if not hmac.compare_digest(
        x_cron_secret,
        expected_secret
    ):

        raise HTTPException(
            status_code=401,
            detail="Token tidak valid."
        )

    # Tolak pemicu jika siklus lain sedang berjalan.
    if cycle_lock.locked():

        raise HTTPException(
            status_code=409,
            detail="Siklus sebelumnya masih berjalan."
        )

    await cycle_lock.acquire()

    try:

        result = await asyncio.to_thread(
            run_ai_cycle
        )

        decision = result.get("ai_result", {})

        return {
            "ok": True,
            "cycle_completed": True,
            "action": decision.get("action", "NONE"),
            "safety_status": decision.get("safety_status"),
            "blocked_reason": decision.get("blocked_reason"),
            "published": result.get("published", False),
            "publish_message": result.get("publish_message", "")
        }

    except Exception:

        logger.exception(
            "[API] Siklus Rule Engine gagal."
        )

        raise HTTPException(
            status_code=500,
            detail="Siklus Rule Engine gagal. Periksa log Render."
        )

    finally:

        cycle_lock.release()
