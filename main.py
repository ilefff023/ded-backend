from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database
import mqtt_client
import ai_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init()
    mqtt_client.start()
    yield


app = FastAPI(title="DED Monitor", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────

class BlinkPayload(BaseModel):
    blink_rate: float = Field(..., ge=0, le=100, description="Blinks per minute")


# ── Helpers ──────────────────────────────────────────────────────────

def _nested_response(rec: dict) -> dict:
    """
    Wraps a flat DB record into the nested structure the frontend expects:
      data.dht22.temperature / data.dht22.humidity
      data.bh1750.lux
      data.mlx90614.obj_temp
    Plus flat fields for convenience.
    """
    t  = rec.get("temp")
    h  = rec.get("humidity")
    l  = rec.get("lux")
    e  = rec.get("eye_temp")
    br = rec.get("blink_rate")
    p  = rec.get("prediction")
    c  = rec.get("confidence")

    # Health score (0–100)
    if p == "severe":
        score = max(10, 40 - round((c or 0.5) * 30))
    elif p == "moderate":
        score = max(40, 65 - round((c or 0.5) * 20))
    else:
        score = min(95, 75 + round((c or 0.5) * 20)) if p else None

    return {
        # Nested sensor fields
        "dht22":    {"temperature": t, "humidity": h},
        "bh1750":   {"lux": l},
        "mlx90614": {"obj_temp": e},
        # Flat aliases
        "temp":       t,
        "humidity":   h,
        "lux":        l,
        "eye_temp":   e,
        "blink_rate": br,
        # AI
        "prediction": p,
        "confidence": c,
        "score":      score,
        "timestamp":  rec.get("timestamp"),
    }


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/blink")
def receive_blink(body: BlinkPayload):
    sensor = mqtt_client.get()

    required = ["temp", "humidity", "lux", "eye_temp"]
    missing  = [f for f in required if f not in sensor]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "MQTT data not yet received",
                "missing": missing,
                "hint":    "Check Mosquitto is running and ESP32 is publishing to 'ded/sensors'"
            }
        )

    temp      = float(sensor["temp"])
    humidity  = float(sensor["humidity"])
    lux       = float(sensor["lux"])
    eye_temp  = float(sensor["eye_temp"])
    blink     = body.blink_rate

    temp_diff  = round(temp - eye_temp, 4)
    blink_norm = round(blink / 60, 6)

    result = ai_model.predict({
        "temp": temp, "humidity": humidity, "lux": lux,
        "eye_temp": eye_temp, "blink_rate": blink
    })

    ts = database.insert(
        temp=temp, humidity=humidity, lux=lux, eye_temp=eye_temp,
        blink_rate=blink, temp_diff=temp_diff, blink_norm=blink_norm,
        prediction=result["prediction"], confidence=result["confidence"]
    )

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "timestamp":  ts,
    }


@app.get("/api/data")
def get_latest():
    # 1 — prefer last saved DB record
    rec = database.latest()
    if rec:
        return _nested_response(rec)

    # 2 — fall back to live MQTT (no prediction yet)
    sensor = mqtt_client.get()
    if sensor:
        return _nested_response({
            "temp":       sensor.get("temp"),
            "humidity":   sensor.get("humidity"),
            "lux":        sensor.get("lux"),
            "eye_temp":   sensor.get("eye_temp"),
            "blink_rate": sensor.get("blink_rate", 0),
            "prediction": None,
            "confidence": None,
        })

    raise HTTPException(
        status_code=503,
        detail={
            "error": "No data available yet",
            "hint":  "Make sure Mosquitto is running and the ESP32 is publishing to 'ded/sensors'"
        }
    )


@app.get("/api/history")
def get_history():
    return [_nested_response(r) for r in database.all_records()]
