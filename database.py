import sqlite3
from datetime import datetime, timezone
from config import DB_NAME


def _conn():
    c = sqlite3.connect(DB_NAME)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                temp        REAL,
                humidity    REAL,
                lux         REAL,
                eye_temp    REAL,
                blink_rate  REAL,
                temp_diff   REAL,
                blink_norm  REAL,
                prediction  TEXT,
                confidence  REAL,
                timestamp   TEXT
            )
        """)
        c.commit()
    print("[DB] ded.db ready")


def insert(temp, humidity, lux, eye_temp, blink_rate,
           temp_diff, blink_norm, prediction, confidence) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("""
            INSERT INTO records
              (temp,humidity,lux,eye_temp,blink_rate,
               temp_diff,blink_norm,prediction,confidence,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (temp, humidity, lux, eye_temp, blink_rate,
              temp_diff, blink_norm, prediction, confidence, ts))
        c.commit()
    return ts


def latest() -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM records ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def all_records() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM records ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]
