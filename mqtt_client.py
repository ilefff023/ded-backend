import json
import threading
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

_lock = threading.Lock()
_latest: dict = {}


def get() -> dict:
    with _lock:
        return dict(_latest)


def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Connected — subscribed to '{MQTT_TOPIC}'")
    else:
        print(f"[MQTT] Connection failed rc={rc}")


def _on_message(client, userdata, msg):
    global _latest
    try:
        payload = json.loads(msg.payload.decode())
        with _lock:
            _latest = payload
        print(f"[MQTT] Received: {payload}")
    except Exception as e:
        print(f"[MQTT] Parse error: {e}")


def start():
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"[MQTT] Could not connect to broker: {e}")
