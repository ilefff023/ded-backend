# database.py — REPLACE ENTIRELY

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "ded.db")

def init():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                dob      TEXT,
                gender   TEXT,
                notes    TEXT,
                created  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id  INTEGER REFERENCES patients(id),
                temp        REAL,
                humidity    REAL,
                lux         REAL,
                eye_temp    REAL,
                blink_rate  REAL,
                temp_diff   REAL,
                blink_norm  REAL,
                prediction  TEXT,
                confidence  REAL,
                cnn_pred    TEXT,
                xgb_pred    TEXT,
                timestamp   TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

def create_patient(name, dob=None, gender=None, notes=None) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, dob, gender, notes) VALUES (?,?,?,?)",
            (name, dob, gender, notes)
        )
        return cur.lastrowid

def get_patients():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM patients ORDER BY created DESC")]

def insert(patient_id=None, **kwargs) -> str:
    ts = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO readings
              (patient_id, temp, humidity, lux, eye_temp, blink_rate,
               temp_diff, blink_norm, prediction, confidence, cnn_pred, xgb_pred, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_id,
            kwargs.get("temp"), kwargs.get("humidity"), kwargs.get("lux"),
            kwargs.get("eye_temp"), kwargs.get("blink_rate"),
            kwargs.get("temp_diff"), kwargs.get("blink_norm"),
            kwargs.get("prediction"), kwargs.get("confidence"),
            kwargs.get("cnn_prediction"), kwargs.get("xgb_prediction"),
            ts
        ))
    return ts

def latest(patient_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if patient_id:
            row = conn.execute(
                "SELECT * FROM readings WHERE patient_id=? ORDER BY timestamp DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

def all_records(patient_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if patient_id:
            rows = conn.execute(
                "SELECT r.*, p.name as patient_name FROM readings r "
                "LEFT JOIN patients p ON r.patient_id = p.id "
                "WHERE r.patient_id=? ORDER BY r.timestamp DESC",
                (patient_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT r.*, p.name as patient_name FROM readings r "
                "LEFT JOIN patients p ON r.patient_id = p.id "
                "ORDER BY r.timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]
