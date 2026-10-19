
# orchestrator_api.py

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_orchestrator import run_ai_cycle, AI_CYCLE_INTERVAL


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def orchestrator_loop():
        print("AI Orchestrator scheduler started.")

        while True:
            try:
                await asyncio.to_thread(run_ai_cycle)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"ORCHESTRATOR ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

            await asyncio.sleep(AI_CYCLE_INTERVAL)

    task = asyncio.create_task(orchestrator_loop())

    try:
        yield
    finally:
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Smart Agriculture AI Orchestrator",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "smart-agriculture-orchestrator"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
