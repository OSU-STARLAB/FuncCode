"""Append-only verification log (runs_hw/verification_log.json).

Every verification-ladder rung (L1-L4) and every accuracy acceptance check
appends one entry here; nothing ever overwrites previous entries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("runs_hw") / "verification_log.json"


def append_verification_entry(entry: dict, log_path=DEFAULT_LOG) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2))
