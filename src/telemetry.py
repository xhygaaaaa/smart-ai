"""轻量埋点与自助解决率

设计：
  - SQLite 单文件（data/telemetry.db），不引入外部服务；
  - 每次问答落一条事件；用户点「已解决/未解决」写回反馈；
  - 同一 session 内、窗口期内的下一次提问，会把上一题标记为 follow_up，
    作为「没有自助解决（用户又问了一遍）」的隐式信号。

自助解决率（文档定义，见 pages/自助解决率.py）：
    solved = 明确点「已解决」 或 窗口期内没有追问且没有负反馈
    rate   = solved / 全部提问
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    session_id         TEXT,
    question           TEXT,
    effective_question TEXT,
    answer             TEXT,
    category           TEXT,
    groundedness       REAL,
    retrieved_count    INTEGER,
    latency_ms         INTEGER,
    retriever          TEXT,
    device             TEXT,
    resolved           INTEGER,   -- 1 已解决 / 0 未解决 / NULL 未反馈
    feedback           TEXT,      -- helpful / unhelpful
    follow_up          INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.TELEMETRY_DB), exist_ok=True)
    conn = sqlite3.connect(config.TELEMETRY_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def log_query(
    session_id: str,
    question: str,
    answer: str,
    category: str,
    groundedness: Optional[float],
    retrieved_count: int,
    latency_ms: int,
    effective_question: Optional[str] = None,
    device: Optional[str] = None,
    retriever: Optional[str] = None,
) -> int:
    """记录一次提问，返回事件 id（用于后续写反馈）。

    同时把同 session、窗口期内上一条未反馈事件标记为 follow_up。
    """
    now = datetime.now()
    cutoff = (now - timedelta(minutes=config.FOLLOWUP_WINDOW_MINUTES)).isoformat(
        timespec="seconds"
    )

    with _conn() as conn:
        conn.execute(
            """UPDATE events SET follow_up = 1
               WHERE id = (
                   SELECT id FROM events
                   WHERE session_id = ? AND resolved IS NULL AND ts >= ?
                   ORDER BY id DESC LIMIT 1
               )""",
            (session_id, cutoff),
        )
        cur = conn.execute(
            """INSERT INTO events
               (ts, session_id, question, effective_question, answer, category,
                groundedness, retrieved_count, latency_ms, retriever, device)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(timespec="seconds"), session_id, question,
             effective_question, answer, category, groundedness,
             retrieved_count, latency_ms, retriever, device),
        )
        return int(cur.lastrowid)


def log_feedback(event_id: int, resolved: bool, feedback: Optional[str] = None):
    """记录用户显式反馈。"""
    with _conn() as conn:
        conn.execute(
            "UPDATE events SET resolved = ?, feedback = ? WHERE id = ?",
            (1 if resolved else 0, feedback, event_id),
        )


def fetch_events(days: int = 7) -> List[Dict]:
    """读取最近 N 天事件（供看板）。"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE ts >= ? ORDER BY id", (cutoff,)
        ).fetchall()
    return [dict(r) for r in rows]


def compute_resolution_rate(events: Optional[List[Dict]] = None,
                            days: int = 7) -> Dict:
    """按文档定义计算自助解决率。"""
    events = fetch_events(days) if events is None else events
    total = len(events)
    if total == 0:
        return {
            "total": 0, "solved": 0, "resolution_rate": None,
            "explicit_feedback": 0, "explicit_solved": 0,
            "explicit_unsolved": 0, "follow_up": 0, "implicit_solved": 0,
            "feedback_coverage": None, "avg_groundedness": None,
            "refusal_rate": None,
        }

    explicit = [e for e in events if e.get("resolved") is not None]
    explicit_solved = [e for e in explicit if e["resolved"] == 1]
    explicit_unsolved = [e for e in explicit if e["resolved"] == 0]

    follow_up = [e for e in events if e.get("resolved") is None and e.get("follow_up")]
    implicit_solved = [e for e in events
                       if e.get("resolved") is None and not e.get("follow_up")]

    solved = len(explicit_solved) + len(implicit_solved)
    grounded = [e["groundedness"] for e in events
                if e.get("groundedness") is not None]
    refused = [e for e in events if e.get("category") == "超纲"]

    return {
        "total": total,
        "solved": solved,
        "resolution_rate": round(solved / total, 4),
        "explicit_feedback": len(explicit),
        "explicit_solved": len(explicit_solved),
        "explicit_unsolved": len(explicit_unsolved),
        "follow_up": len(follow_up),
        "implicit_solved": len(implicit_solved),
        "feedback_coverage": round(len(explicit) / total, 4),
        "avg_groundedness": round(sum(grounded) / len(grounded), 4) if grounded else None,
        "refusal_rate": round(len(refused) / total, 4),
    }
