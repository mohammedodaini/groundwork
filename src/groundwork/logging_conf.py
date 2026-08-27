"""Logging configuration.

Two things worth defending in an interview:

1. JSON output in production, human-readable locally. Log aggregators need
   structured events; developers need to read them.
2. A redaction filter. An agent handles API keys and fetched web content; the
   easiest way to leak a key is to log a request object. The filter is a
   backstop, not a licence to be careless.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

from groundwork.config import Settings

_SECRET_PATTERNS = [
    re.compile(r"(sk-ant-[A-Za-z0-9\-_]{8,})"),
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
    re.compile(r"(tvly-[A-Za-z0-9\-_]{8,})"),
    re.compile(r"((?:api[_-]?key|token|secret|password)\"?\s*[:=]\s*\"?)([^\s\",}]{6,})"),
]

_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
    "taskName",
}


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(r"\1***REDACTED***", text)
        else:
            text = pattern.sub("***REDACTED***", text)
    return text


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, including any `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False))


class HumanFormatter(logging.Formatter):
    """Compact local format: `LEVEL logger event key=value`."""

    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        )
        base = f"{record.levelname:<7} {record.name:<34} {record.getMessage()}"
        line = f"{base}  {extras}" if extras else base
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return redact(line)


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else HumanFormatter())
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    # These are noisy and rarely useful at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
