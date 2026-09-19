import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,          -- "rag_query", "tool_call", "agent_run"
    claim_id TEXT,
    question TEXT,
    answer TEXT,
    latency_seconds REAL,
    avg_retrieval_score REAL,
    extra_json TEXT                    -- free-form JSON for anything else
);
"""


@contextmanager
def _connect():
    """Context manager for a SQLite connection so we always close it cleanly."""
    conn = sqlite3.connect(settings.sqlite_db_path)
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_event(
    event_type: str,
    claim_id: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    latency_seconds: Optional[float] = None,
    avg_retrieval_score: Optional[float] = None,
    extra_json: Optional[str] = None,
) -> None:
    """Writes one row to the query_log table. Never raises -- logging failures
    should never break the main app."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO query_log
                (timestamp, event_type, claim_id, question, answer,
                 latency_seconds, avg_retrieval_score, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    event_type,
                    claim_id,
                    question,
                    answer,
                    latency_seconds,
                    avg_retrieval_score,
                    extra_json,
                ),
            )
    except Exception as e:
        print(f"[logging warning] Failed to log event: {e}")


def fetch_recent_logs(limit: int = 50):
    """Returns the most recent log rows as a list of dicts (used by the
    Streamlit Evaluation tab)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


def compute_summary_stats():
    """Aggregate stats for the evaluation report: avg latency, avg retrieval
    score, count of queries per event type."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM query_log").fetchall()

    if not rows:
        return {
            "total_events": 0,
            "avg_latency_seconds": None,
            "avg_retrieval_score": None,
            "by_event_type": {},
        }

    latencies = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None]
    scores = [r["avg_retrieval_score"] for r in rows if r["avg_retrieval_score"] is not None]

    by_type = {}
    for r in rows:
        by_type[r["event_type"]] = by_type.get(r["event_type"], 0) + 1

    return {
        "total_events": len(rows),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "avg_retrieval_score": sum(scores) / len(scores) if scores else None,
        "by_event_type": by_type,
    }


if __name__ == "__main__":
    # Manual test: python -m src.utils
    log_event(event_type="rag_query", claim_id="CLM-1001", question="test?",
              answer="test answer", latency_seconds=1.23, avg_retrieval_score=0.81)
    print(compute_summary_stats())
    print(fetch_recent_logs(limit=5))
