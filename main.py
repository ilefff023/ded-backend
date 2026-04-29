from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import os

import database
import ai_model
import mqtt_client

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. MODÈLES DE DONNÉES
class PatientCreate(BaseModel):
    name: str
    dob: Optional[str] = None
    gender: Optional[str] = None
    notes: Optional[str] = None

class BlinkRequest(BaseModel):
    blink_rate: float = Field(..., ge=0, le=100)
    patient_id: Optional[int] = None  # Ajouté pour correspondre à votre logique d'insertion

# 2. CONFIGURATION DE L'APP (Doit être AVANT les routes)
@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init()
    ai_model.load_model()
    mqtt_client.start()
    yield

app = FastAPI(title="DED Monitor API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. FONCTIONS UTILITAIRES
def build_response(temp, humidity, lux, eye_temp, blink_rate,
                   prediction=None, confidence=None, score=None,
                   timestamp=None):
    if score is None and prediction:
        if prediction == "severe":
            score = max(10, 40 - round((confidence or 0.5) * 30))
        elif prediction == "moderate":
            score = max(40, 65 - round((confidence or 0.5) * 20))
        else:
            score = min(95, 75 + round((confidence or 0.5) * 20))

    return {
        "dht22":    {"temperature": round(temp, 1)     if temp     is not None else None,
                     "humidity":    round(humidity, 1) if humidity is not None else None},
        "bh1750":   {"lux":         round(lux, 1)      if lux      is not None else None},
        "mlx90614": {"obj_temp":    round(eye_temp, 2) if eye_temp is not None else None},
        "temp":       round(temp, 1)       if temp       is not None else None,
        "humidity":   round(humidity, 1)   if humidity   is not None else None,
        "lux":        round(lux, 1)        if lux        is not None else None,
        "eye_temp":   round(eye_temp, 2)   if eye_temp   is not None else None,
        "blink_rate": round(blink_rate, 1) if blink_rate is not None else None,
        "prediction": prediction,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "score":      score,
        "timestamp":  timestamp or datetime.utcnow().isoformat(),
    }

# 4. ROUTES API
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/patients")
def create_patient(body: PatientCreate):
    pid = database.create_patient(body.name, body.dob, body.gender, body.notes)
    return {"id": pid, "name": body.name}

@app.get("/api/patients")
def list_patients():
    return database.get_patients()

@app.get("/api/patients/{patient_id}/history")
def patient_history(patient_id: int):
    # Assurez-vous que _nested_response est définie dans database ou ici
    return [r for r in database.all_records(patient_id)]

@app.post("/api/blink")
def receive_blink(body: BlinkRequest):
    sensor = mqtt_client.get()
    missing = [f for f in ["temp", "humidity", "lux", "eye_temp"] if f not in sensor]
    if missing:
        raise HTTPException(status_code=503, detail={"error": "MQTT data missing", "missing": missing})

    temp, humidity, lux, eye_temp = (float(sensor[k]) for k in ["temp","humidity","lux","eye_temp"])
    blink = body.blink_rate

    result = ai_model.predict({"temp": temp, "humidity": humidity, "lux": lux, "eye_temp": eye_temp, "blink_rate": blink})
    
    # Correction : Calcul ou suppression des variables manquantes (temp_diff, blink_norm)
    ts = database.insert(
        patient_id=body.patient_id,
        temp=temp, humidity=humidity, lux=lux, eye_temp=eye_temp,
        blink_rate=blink,
        prediction=result["prediction"], 
        confidence=result["confidence"]
    )
    return build_response(temp, humidity, lux, eye_temp, blink,
                          prediction=result["prediction"], confidence=result["confidence"], timestamp=ts)

@app.get("/api/data")
def get_latest():
    record = database.latest()
    if record:
        return build_response(
            temp=record["temp"], humidity=record["humidity"],
            lux=record["lux"],   eye_temp=record["eye_temp"],
            blink_rate=record.get("blink_rate"),
            prediction=record.get("prediction"), confidence=record.get("confidence"),
            timestamp=record.get("timestamp"),
        )
    raise HTTPException(status_code=503, detail="No data available")

