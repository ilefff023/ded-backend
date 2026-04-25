"""
AI prediction module.
Currently uses rule-based heuristics (TFOS DEWS II thresholds).
Replace the body of predict() with joblib.load / model.predict
when your trained model is ready — the interface stays identical.
"""


def predict(data: dict) -> dict:
    temp       = float(data.get("temp",       0))
    humidity   = float(data.get("humidity",   0))
    lux        = float(data.get("lux",        0))
    eye_temp   = float(data.get("eye_temp",   0))
    blink_rate = float(data.get("blink_rate", 0))

    risk = 0.0

    # Humidity
    if humidity < 30:    risk += 35
    elif humidity < 40:  risk += 20
    elif humidity < 50:  risk += 8

    # Luminosity
    if lux > 700:        risk += 25
    elif lux > 500:      risk += 12

    # Eye temperature (normal 34–35.5 °C)
    if eye_temp < 33.5:  risk += 30
    elif eye_temp < 34:  risk += 15
    elif eye_temp > 36:  risk += 8

    # Blink rate (normal 12–20 /min)
    if blink_rate < 8:   risk += 20
    elif blink_rate < 12: risk += 10
    elif blink_rate > 25: risk += 5

    # Ambient temperature
    if temp > 28 or temp < 18: risk += 5

    if risk >= 55:
        return {"prediction": "severe",   "confidence": round(min(0.65 + risk / 200, 0.97), 4)}
    elif risk >= 28:
        return {"prediction": "moderate", "confidence": round(min(0.55 + risk / 200, 0.90), 4)}
    else:
        return {"prediction": "normal",   "confidence": round(min(0.70 + (55 - risk) / 200, 0.97), 4)}
