import config
from src import telemetry


def _log(session="s1", question="q"):
    return telemetry.log_query(
        session_id=session,
        question=question,
        answer="a",
        category="有效",
        groundedness=0.8,
        retrieved_count=3,
        latency_ms=120,
    )


def test_followup_marks_previous_event(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", str(tmp_path / "t.db"))
    _log(question="第一问")
    _log(question="追问")

    events = telemetry.fetch_events(days=1)
    assert len(events) == 2
    assert events[0]["follow_up"] == 1
    assert events[1]["follow_up"] == 0


def test_different_session_not_followup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", str(tmp_path / "t.db"))
    _log(session="s1")
    _log(session="s2")
    events = telemetry.fetch_events(days=1)
    assert all(e["follow_up"] == 0 for e in events)


def test_resolution_rate_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", str(tmp_path / "t.db"))
    _log(question="第一问")
    _log(question="追问")

    summary = telemetry.compute_resolution_rate(days=1)
    assert summary["total"] == 2
    # 第一问被追问=未解决；第二问未反馈且无追问=隐式已解决
    assert summary["follow_up"] == 1
    assert summary["resolution_rate"] == 0.5

    second_id = telemetry.fetch_events(days=1)[-1]["id"]
    telemetry.log_feedback(second_id, resolved=False, feedback="unhelpful")
    assert telemetry.compute_resolution_rate(days=1)["resolution_rate"] == 0.0


def test_explicit_solved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", str(tmp_path / "t.db"))
    event_id = _log()
    telemetry.log_feedback(event_id, resolved=True, feedback="helpful")
    summary = telemetry.compute_resolution_rate(days=1)
    assert summary["resolution_rate"] == 1.0
    assert summary["feedback_coverage"] == 1.0


def test_empty_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", str(tmp_path / "empty.db"))
    summary = telemetry.compute_resolution_rate(days=1)
    assert summary["total"] == 0
    assert summary["resolution_rate"] is None
