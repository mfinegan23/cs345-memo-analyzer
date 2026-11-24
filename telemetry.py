"""Telemetry for Customer Call Reviewer (Application Insights via OpenTelemetry).

Usage in app.py, immediately after load_dotenv() and BEFORE importing Flask or
any Azure SDK:

    from telemetry import init_telemetry
    init_telemetry()

PRIVACY RULE: nothing in this file ever receives transcript text. Spans, metrics
and the session log carry counts, labels, durations and flags only. Exceptions
are recorded by type name, never by message.
"""
import math
import os
import threading
import time
from contextlib import contextmanager

_tracer = None
_m = {}  # metric instruments, created after the provider is configured


def init_telemetry():
    """Configure Azure Monitor, then create tracer and metric instruments."""
    global _tracer
    if _tracer is not None:
        return
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING not set")

    from azure.monitor.opentelemetry import configure_azure_monitor
    configure_azure_monitor(connection_string=conn)

    from opentelemetry import metrics, trace
    _tracer = trace.get_tracer("call-reviewer")
    meter = metrics.get_meter("call-reviewer")

    # Gauges: one value per call (as in the handout)
    for name in ("stt_confidence", "stt_word_count", "language_entity_count",
                 "language_keyphrase_count", "language_sentiment"):
        _m[name] = meter.create_gauge(name)

    # Histograms: keep sum and count, so no call is lost when several land in
    # one export interval. Needed for the cost model (sums) and for
    # call_flagged (average == share flagged).
    for name, unit in (("stt_duration_seconds", "s"), ("tts_char_count", "{char}"),
                       ("language_text_chars", "{char}"),
                       ("pii_redacted_count", "{item}"), ("call_flagged", "1"),
                       ("stage_stt_ms", "ms"), ("stage_pii_ms", "ms"),
                       ("stage_language_ms", "ms"), ("stage_tts_ms", "ms")):
        _m[name] = meter.create_histogram(name, unit=unit)
    print("Application Insights initialized.")


def current_span():
    from opentelemetry import trace
    return trace.get_current_span()


@contextmanager
def timed_span(name, attrs=None):
    """Span plus wall-clock timing. Yields {'ms': float, 'span': span}.
    Exceptions are marked on the span by type only (no message)."""
    from opentelemetry.trace import Status, StatusCode
    holder = {"ms": 0.0, "span": None}
    t0 = time.perf_counter()
    with _tracer.start_as_current_span(
            name, record_exception=False, set_status_on_exception=False) as span:
        holder["span"] = span
        for k, v in (attrs or {}).items():
            span.set_attribute(k, v)
        try:
            yield holder
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            holder["ms"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("duration_ms", round(holder["ms"], 1))


# ------------------------------------------------------------------ metrics

SENTIMENT_NUM = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def flag_reason_label(reasons):
    return ",".join(reasons) if reasons else "none"


def record_pipeline_metrics(c):
    """c: dict of counts, labels and timings for one completed call."""
    attrs = {"audio_format": c["audio_format"], "language": c["language"]}
    _m["stt_confidence"].set(c["confidence"], attrs)
    _m["stt_word_count"].set(c["word_count"], attrs)
    _m["language_entity_count"].set(c["entity_count"], attrs)
    _m["language_keyphrase_count"].set(c["keyphrase_count"], attrs)
    _m["language_sentiment"].set(SENTIMENT_NUM.get(c["sentiment"], 0.0), attrs)

    _m["stt_duration_seconds"].record(c["duration_seconds"], attrs)
    _m["tts_char_count"].record(c["tts_chars"], attrs)
    _m["language_text_chars"].record(c["text_chars"], attrs)
    _m["pii_redacted_count"].record(c["pii_count"], attrs)
    _m["stage_stt_ms"].record(c["stt_ms"], attrs)
    _m["stage_pii_ms"].record(c["pii_ms"], attrs)
    _m["stage_language_ms"].record(c["language_ms"], attrs)
    _m["stage_tts_ms"].record(c["tts_ms"], attrs)

    # Domain metric: 1 if flagged, else 0; reason as an attribute
    _m["call_flagged"].record(
        1 if c["flagged"] else 0,
        {**attrs, "flag_reason": flag_reason_label(c["reasons"])})


def emit_pipeline_completed(span, c):
    """Event as attributes on the request span (shows in requests.customDimensions)."""
    span.set_attribute("event.name", "pipeline_completed")
    span.set_attribute("stt.confidence", c["confidence"])
    span.set_attribute("stt.language", c["language"])
    span.set_attribute("entities.count", c["entity_count"])
    span.set_attribute("sentiment", c["sentiment"])
    span.set_attribute("audio.format", c["audio_format"])
    span.set_attribute("pii.redacted_count", c["pii_count"])
    span.set_attribute("call.flagged", bool(c["flagged"]))
    span.set_attribute("call.flag_reason", flag_reason_label(c["reasons"]))


def emit_pipeline_error(span, stage, error_type, audio_format):
    span.set_attribute("event.name", "pipeline_error")
    span.set_attribute("error.stage", stage)
    span.set_attribute("error.type", error_type)
    span.set_attribute("audio.format", audio_format)


# --------------------------------------------------------------- session log

class SessionLog:
    """In-memory log of per-call metrics. Holds NO transcript text."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows = []

    def add(self, c):
        row = {
            "confidence": c["confidence"], "language": c["language"],
            "duration_seconds": c["duration_seconds"],
            "entity_count": c["entity_count"], "sentiment": c["sentiment"],
            "pii_count": c["pii_count"], "flagged": 1 if c["flagged"] else 0,
            "flag_reason": flag_reason_label(c["reasons"]),
            "tts_chars": c["tts_chars"],
            "stt_ms": round(c["stt_ms"], 1), "language_ms": round(c["language_ms"], 1),
            "tts_ms": round(c["tts_ms"], 1),
        }
        with self._lock:
            self._rows.append(row)

    @staticmethod
    def _p95(values):
        s = sorted(values)
        return s[max(0, math.ceil(0.95 * len(s)) - 1)]

    def summary(self):
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return {"message": "No calls yet"}
        n = len(rows)
        confs = [r["confidence"] for r in rows]
        flagged = sum(r["flagged"] for r in rows)
        reasons = {}
        for r in rows:
            if r["flagged"]:
                for part in r["flag_reason"].split(","):
                    reasons[part] = reasons.get(part, 0) + 1
        return {
            "total_calls": n,
            "avg_confidence": round(sum(confs) / n, 3),
            "min_confidence": round(min(confs), 3),
            "flagged_calls": flagged,
            "flagged_share": round(flagged / n, 3),
            "flag_reasons": reasons,
            "pii_items_redacted": sum(r["pii_count"] for r in rows),
            "p95_stt_ms": round(self._p95([r["stt_ms"] for r in rows]), 1),
            "avg_stage_ms": {
                "stt": round(sum(r["stt_ms"] for r in rows) / n, 1),
                "language": round(sum(r["language_ms"] for r in rows) / n, 1),
                "tts": round(sum(r["tts_ms"] for r in rows) / n, 1),
            },
            "total_audio_seconds": round(sum(r["duration_seconds"] for r in rows), 1),
            "total_tts_chars": sum(r["tts_chars"] for r in rows),
            "sentiment_breakdown": {
                s: sum(1 for r in rows if r["sentiment"] == s)
                for s in ("positive", "neutral", "negative")},
            "calls": rows[-10:],
        }


session = SessionLog()
