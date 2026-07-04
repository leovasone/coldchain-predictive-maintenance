"""FastAPI app: simulates a fleet of cold-chain refrigeration assets spread
across several Latin American cities, scores each one's telemetry with two
trained ML models, escalates likely failures through two Claude agents
(diagnosis, then dispatch), and streams everything to the browser over
WebSockets in real time.

Entirely synthetic end to end -- there is no real fleet, no real IoT
hardware, and no real technicians behind this. See README "Honest notes"
for what that does and doesn't mean for the ML models' reported accuracy.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import diagnostics_agent, dispatch_agent, ml_models
from .locations import build_fleet, build_technicians
from .telemetry_sim import AssetState, next_reading

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coldchain-predictive-maintenance")

POLL_INTERVAL_SECONDS = 12
READING_WINDOW = 8  # must match train_models.py's SEQ_LEN
FAILURE_PROBABILITY_ALERT_THRESHOLD = 0.30
# Real dispatch ETAs are 10-100+ minutes; compressing them into a handful of
# demo-seconds keeps a live viewer from having to wait real minutes to see
# an alert resolve. This is a UI/demo pacing choice, documented in the
# README, not a claim about real repair times.
SIM_SECONDS_PER_ETA_MINUTE = 2.5

app = FastAPI(title="Cold-Chain Predictive Maintenance")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ConnectionManager:
    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, message: dict):
        try:
            await ws.send_json(message)
        except Exception:
            pass


manager = ConnectionManager()

FLEET = {a["id"]: a for a in build_fleet()}
TECHNICIANS = {t["id"]: t for t in build_technicians()}
ASSET_STATES = {aid: AssetState(asset_id=aid, asset_type=a["asset_type"]) for aid, a in FLEET.items()}
READING_HISTORY: dict[str, list] = {aid: [] for aid in FLEET}
LATEST_SCORE: dict[str, dict] = {}  # asset_id -> last ml_models.score() output
ACTIVE_ALERTS: dict[int, dict] = {}  # alert_id -> alert dict
_alert_id_counter = 0
_rng = random.Random(42)


@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "hint": "frontend not bundled in this deployment"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "clients": len(manager.active),
        "fleet_size": len(FLEET),
        "technicians": len(TECHNICIANS),
        "active_alerts": len(ACTIVE_ALERTS),
        "ml_models_available": ml_models.models_available(),
    }


@app.get("/ml-metadata")
async def ml_metadata():
    """Static-ish info about the two trained models, for the frontend's
    'ML insights' panel: model type, held-out ROC-AUC, feature importances,
    and an explicit note that training data is synthetic."""
    return ml_models.metadata() or {"error": "models not trained -- run python -m backend.train_models"}


def _fleet_snapshot() -> dict:
    return {
        "type": "snapshot",
        "fleet": list(FLEET.values()),
        "technicians": list(TECHNICIANS.values()),
        "active_alerts": list(ACTIVE_ALERTS.values()),
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send the current full state immediately so a client that connects
    # mid-run doesn't have to wait for the next poll cycle to see anything
    # -- same fix applied to the realtime-weather-insights project after a
    # client reported staring at an empty panel for hours.
    await manager.send_to(websocket, _fleet_snapshot())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _available_technicians() -> list[dict]:
    return [t for t in TECHNICIANS.values() if t["status"] == "available"]


async def _raise_alert(asset: dict, readings: list, ml_result: dict):
    global _alert_id_counter
    diag = diagnostics_agent.diagnose(asset, readings, ml_result)
    disp = dispatch_agent.dispatch(
        diag["alert_type"], diag["severity"], diag["diagnosis"], asset, _available_technicians()
    )

    _alert_id_counter += 1
    alert_id = _alert_id_counter
    alert = {
        "id": alert_id,
        "asset_id": asset["id"],
        "asset_name": asset["name"],
        "city": asset["city"],
        "country": asset["country"],
        "lat": asset["lat"],
        "lon": asset["lon"],
        "alert_type": diag["alert_type"],
        "severity": diag["severity"],
        "diagnosis": diag["diagnosis"],
        "recommended_action": diag["recommended_action"],
        "diagnosis_source": diag["source"],
        "ml": ml_result,
        "dispatch": disp,
        "status": "dispatched" if disp.get("technician_id") else "queued",
    }
    ACTIVE_ALERTS[alert_id] = alert
    asset["status"] = "critical" if diag["severity"] == "critical" else "warning"

    if disp.get("technician_id") is not None:
        TECHNICIANS[disp["technician_id"]]["status"] = "dispatched"
        eta = disp.get("eta_min") or 20
        resolve_delay = eta * SIM_SECONDS_PER_ETA_MINUTE
        asyncio.create_task(_resolve_after(alert_id, resolve_delay))

    await manager.broadcast({"type": "alert", "alert": alert})
    log.info("alert %d raised for %s (%s/%s, source=%s)", alert_id, asset["id"], diag["alert_type"], diag["severity"], diag["source"])


async def _resolve_after(alert_id: int, delay_seconds: float):
    await asyncio.sleep(delay_seconds)
    alert = ACTIVE_ALERTS.pop(alert_id, None)
    if alert is None:
        return
    asset = FLEET.get(alert["asset_id"])
    if asset is not None:
        asset["status"] = "ok"
    tech_id = alert["dispatch"].get("technician_id")
    if tech_id is not None and tech_id in TECHNICIANS:
        TECHNICIANS[tech_id]["status"] = "available"
    await manager.broadcast({
        "type": "alert_resolved",
        "alert_id": alert_id,
        "asset_id": alert["asset_id"],
        "technician_id": tech_id,
    })
    log.info("alert %d resolved (asset %s back to ok)", alert_id, alert["asset_id"])


async def poll_once():
    for asset_id, asset in FLEET.items():
        state = ASSET_STATES[asset_id]
        reading = next_reading(state, _rng)

        history = READING_HISTORY[asset_id]
        history.append(reading)
        if len(history) > READING_WINDOW:
            history.pop(0)

        score = None
        if len(history) >= READING_WINDOW and ml_models.models_available():
            score = ml_models.score(history, asset["asset_type"])
            LATEST_SCORE[asset_id] = score

        await manager.broadcast({
            "type": "reading",
            "asset_id": asset_id,
            "temperature_c": reading.temperature_c,
            "current_a": reading.current_a,
            "duty_cycle_pct": reading.compressor_duty_cycle_pct,
            "door_events_last_hour": reading.door_events_last_hour,
            "anomaly_score": score["anomaly_score"] if score else None,
            "failure_probability": score["failure_probability"] if score else None,
            "status": asset["status"],
        })

        already_flagged = any(a["asset_id"] == asset_id for a in ACTIVE_ALERTS.values())
        if score and not already_flagged and score["failure_probability"] >= FAILURE_PROBABILITY_ALERT_THRESHOLD:
            await _raise_alert(asset, list(history), score)


async def poll_loop():
    while True:
        await poll_once()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_task():
    asyncio.create_task(poll_loop())
