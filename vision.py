"""
vision.py  —  DED Monitor Eye Tracking
Runs your best_model.h5 CNN on the ESP32-CAM stream,
detects blinks, and exposes results via GET /api/vision
so the frontend can display them in real time.

Run independently:
    python vision.py

FastAPI (main.py) must also be running on port 8000.
"""

import os, sys, time, threading, queue, urllib.request, json, sqlite3
import numpy as np

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import h5py, re

# ══════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════
ESP32_IP       = "172.20.10.2"           # ← ESP32-CAM IP from Serial Monitor
STREAM_URL     = f"http://{ESP32_IP}/stream"
BACKEND_URL    = "http://localhost:8000"  # FastAPI endpoint
MODEL_PATH     = "best_model.h5"
IMG_SIZE       = 80

# ══════════════════════════════════════════════════
#  SHARED STATE  (read by FastAPI via /api/vision)
# ══════════════════════════════════════════════════
_state_lock = threading.Lock()
_state = {
    "eye_state":   "Initialisation",
    "blink_count": 0,
    "blink_rate":  0.0,
    "threshold":   0.50,
    "running":     False,
    "stream_ok":   False,
    "timestamp":   None,
}

def get_state() -> dict:
    with _state_lock:
        return dict(_state)

def set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)
        _state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")


# ══════════════════════════════════════════════════
#  MJPEG STREAM READER
# ══════════════════════════════════════════════════
class MJPEGCapture:
    def __init__(self, url):
        self.url    = url
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop:
            try:
                print(f"[Stream] Connecting to {self.url} ...")
                stream = urllib.request.urlopen(self.url, timeout=15)
                set_state(stream_ok=True)
                print("[Stream] Connected!")
                buf = b""
                while not self._stop:
                    chunk = stream.read(4096)
                    if not chunk: break
                    buf += chunk
                    a  = buf.find(b'\xff\xd8')
                    b_ = buf.find(b'\xff\xd9')
                    if a != -1 and b_ != -1 and b_ > a:
                        jpg = buf[a:b_+2]; buf = buf[b_+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
            except Exception as e:
                set_state(stream_ok=False)
                print(f"[Stream] Lost: {e}. Retrying in 3s...")
                time.sleep(3)

    def read(self):
        with self._lock:
            return (True, self._frame.copy()) if self._frame is not None else (False, None)

    def isOpened(self): return not self._stop
    def release(self):  self._stop = True
    def set(self, *a, **k): pass


# ══════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════
def build_model():
    m = models.Sequential([
        layers.Conv2D(32,(3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    return m

def load_model_safe():
    keras_path = MODEL_PATH.replace('.h5', '.keras')
    for path, loader in [
        (keras_path, lambda p: keras.models.load_model(p, compile=False)),
        (MODEL_PATH,  lambda p: keras.models.load_model(p, compile=False)),
    ]:
        if os.path.exists(path):
            try:
                m = loader(path)
                print(f"[Model] Loaded {path}")
                return m
            except Exception as e:
                print(f"[Model] {path} failed: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            m = build_model()
            m.load_weights(MODEL_PATH)
            print("[Model] Weights loaded into rebuilt model")
            return m
        except Exception as e:
            print(f"[Model] Weights load failed: {e}")

    print("[Model] ERROR: could not load model. Place best_model.h5 next to vision.py")
    sys.exit(1)


# ══════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

def preprocess(eye_gray):
    if eye_gray is None or eye_gray.size == 0 or min(eye_gray.shape[:2]) < 5:
        return None
    eye = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
    eye = CLAHE.apply(eye)
    eye = eye.astype(np.float32) / 255.0
    return eye.reshape(IMG_SIZE, IMG_SIZE, 1)

def get_eye_crops(face_gray):
    fh, fw = face_gray.shape[:2]
    y1, y2   = int(fh*0.18), int(fh*0.52)
    lx1, lx2 = int(fw*0.04), int(fw*0.46)
    rx1, rx2 = int(fw*0.54), int(fw*0.96)
    return face_gray[y1:y2, lx1:lx2], face_gray[y1:y2, rx1:rx2]


# ══════════════════════════════════════════════════
#  PUSH BLINK DATA TO FASTAPI
# ══════════════════════════════════════════════════
def push_blink_to_backend(blink_rate: float):
    """POST /api/blink so the main backend stores the data and runs AI."""
    try:
        payload = json.dumps({"blink_rate": round(blink_rate, 2)}).encode()
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/blink",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        pass  # Backend may not have MQTT data yet — silent fail


# ══════════════════════════════════════════════════
#  MAIN TRACKING LOOP
# ══════════════════════════════════════════════════
def run_tracking():
    model = load_model_safe()
    dummy = np.zeros((2, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("[Model] Warmed up.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Async inference queue
    infer_q  = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)

    def infer_worker():
        while True:
            eyes = infer_q.get()
            if eyes is None: break
            processed = [preprocess(e) for e in eyes]
            processed = [p for p in processed if p is not None]
            avg = float(np.mean(model.predict(np.stack(processed), verbose=0))) if processed else None
            try: result_q.get_nowait()
            except: pass
            result_q.put(avg)

    threading.Thread(target=infer_worker, daemon=True).start()

    cap = MJPEGCapture(STREAM_URL)
    print("[Vision] Waiting for stream (30s)...")
    for i in range(150):
        ret, _ = cap.read()
        if ret: break
        if i % 20 == 0 and i > 0: print(f"  {i*0.2:.0f}s...")
        time.sleep(0.2)

    ret, _ = cap.read()
    if not ret:
        print(f"[Vision] ERROR: No frames from {STREAM_URL}")
        print("  Open that URL in a browser to verify the stream works.")
        return

    print("[Vision] Stream OK — tracking started.")
    set_state(running=True, stream_ok=True)

    OPEN_THRESHOLD = 0.50
    blink_count    = 0
    start_time     = time.time()
    last_blink     = 0.0
    eye_closed     = False
    blink_start    = 0.0
    pred_buf       = []
    last_face      = None
    last_raw       = None
    frame_count    = 0
    last_push      = time.time()
    last_db_write  = time.time()

    # SQLite for local logging
    conn   = sqlite3.connect("eye_data.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS eye_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, eye_state TEXT, blink INTEGER, blink_rate REAL)""")
    conn.commit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame      = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq    = cv2.equalizeHist(gray)
        frame_count += 1
        blink_event  = 0
        eye_found    = False

        # Face detection every 4 frames
        if frame_count % 4 == 0:
            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
            if len(faces) > 0:
                last_face = max(faces, key=lambda r: r[2]*r[3])

        if last_face is not None:
            fx,fy,fw,fh = last_face
            fx,fy = max(0,fx), max(0,fy)
            fw,fh = min(fw,frame_w-fx), min(fh,frame_h-fy)
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            lg, rg = get_eye_crops(face_gray)

            if infer_q.empty():
                infer_q.put([lg, rg])
            try:
                last_raw = result_q.get_nowait()
            except queue.Empty:
                pass

            if last_raw is not None:
                eye_found = True
                pred_buf.append(last_raw)
                if len(pred_buf) > 4: pred_buf.pop(0)
                avg_pred = float(np.mean(pred_buf))
                label    = "Open" if avg_pred >= OPEN_THRESHOLD else "Closed"

                # Blink logic
                current_time = time.time()
                if label == "Closed":
                    if not eye_closed:
                        eye_closed  = True
                        blink_start = current_time
                else:
                    if eye_closed:
                        dur = current_time - blink_start
                        if dur > 0.05 and (current_time - last_blink) > 0.25:
                            blink_count += 1
                            blink_event  = 1
                            last_blink   = current_time
                        eye_closed = False

                elapsed    = time.time() - start_time
                blink_rate = (blink_count / elapsed) * 60.0 if elapsed > 0 else 0.0

                set_state(
                    eye_state   = label,
                    blink_count = blink_count,
                    blink_rate  = round(blink_rate, 1),
                    threshold   = OPEN_THRESHOLD,
                )

                # Push to backend every 3s
                if time.time() - last_push > 3:
                    threading.Thread(target=push_blink_to_backend,
                                     args=(blink_rate,), daemon=True).start()
                    last_push = time.time()

                # DB write every 0.5s
                if time.time() - last_db_write > 0.5:
                    cursor.execute(
                        "INSERT INTO eye_tracking (timestamp,eye_state,blink,blink_rate) VALUES (?,?,?,?)",
                        (time.strftime("%Y-%m-%d %H:%M:%S"), label, blink_event, round(blink_rate,2))
                    )
                    conn.commit()
                    last_db_write = time.time()

        else:
            eye_closed = False
            pred_buf.clear()
            set_state(eye_state="Aucun visage")

        # Display overlay
        elapsed    = time.time() - start_time
        blink_rate = (blink_count / elapsed) * 60.0 if elapsed > 0 else 0.0
        state_c    = (0,255,0) if eye_found and _state["eye_state"]=="Open" else (0,0,255)
        cv2.putText(frame, f"State: {_state['eye_state']}",
                    (20,40),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_c, 2)
        cv2.putText(frame, f"Blinks: {blink_count} | Rate: {int(blink_rate)}/min",
                    (20,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"ESP32: {ESP32_IP}",
                    (20, frame_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
        cv2.imshow("DED Monitor — Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    infer_q.put(None)
    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    set_state(running=False)
    print(f"\n[Vision] Done. Blinks: {blink_count} | Rate: {blink_rate:.1f}/min")


# ══════════════════════════════════════════════════
#  /api/vision ENDPOINT  (mini HTTP server on port 8001)
#  FastAPI on 8000 will proxy this internally.
#  OR just import get_state() directly if merged.
# ══════════════════════════════════════════════════
from http.server import BaseHTTPRequestHandler, HTTPServer

class VisionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/vision'):
            data = json.dumps(get_state()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args): pass  # suppress access logs


def start_vision_server():
    server = HTTPServer(('0.0.0.0', 8001), VisionHandler)
    print("[VisionAPI] Running on http://localhost:8001/api/vision")
    server.serve_forever()


if __name__ == '__main__':
    # Start mini API server in background
    threading.Thread(target=start_vision_server, daemon=True).start()
    # Run tracking (blocking)
    run_tracking()
