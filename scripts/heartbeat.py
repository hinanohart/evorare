#!/usr/bin/env python3
"""Update session_lock.last_heartbeat_utc in the progress file (stale-lock takeover support)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROG = Path(__file__).resolve().parent.parent / ".evorare-progress.json"


def main() -> int:
    if not PROG.exists():
        return 0
    data = json.loads(PROG.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock = data.setdefault("session_lock", {})
    lock["pid"] = os.getpid()
    if not lock.get("started_at_utc"):
        lock["started_at_utc"] = now
    lock["last_heartbeat_utc"] = now
    PROG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"heartbeat {now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
