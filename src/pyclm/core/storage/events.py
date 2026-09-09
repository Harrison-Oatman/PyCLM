"""
The events table: what changed during a run, and what went wrong.

One row per applied or refused setting request, per late timepoint and per
acquisition error, written by the Manager to ``events.parquet`` in the
experiment directory (rewritten on every row) and to ``events.csv`` at
close. ``pyclm.io`` exposes it as ``ExperimentData.events``.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

EVENT_COLUMNS = (
    "wall_time",
    "t_requested",
    "t_applied",
    "experiment",
    "channel",
    "kind",
    "key",
    "old",
    "new",
    "status",
    "detail",
    "source",
)
EVENT_SCHEMA = pa.schema(
    [
        ("wall_time", pa.string()),
        ("t_requested", pa.int32()),
        ("t_applied", pa.int32()),
        ("experiment", pa.string()),
        ("channel", pa.string()),
        ("kind", pa.string()),
        ("key", pa.string()),
        ("old", pa.string()),
        ("new", pa.string()),
        ("status", pa.string()),
        ("detail", pa.string()),
        ("source", pa.string()),
    ]
)


def _text(value) -> str | None:
    return None if value is None else str(value)


class EventLog:
    """Append-only record of runtime events, persisted as parquet (and CSV at close)."""

    def __init__(self, path: Path | None):
        self.path = None if path is None else Path(path)
        self.rows: list[dict] = []
        self._lock = threading.Lock()
        self.closed = False

    def record(
        self,
        kind: str,
        experiment: str | None = None,
        channel: str | None = None,
        key: str | None = None,
        old=None,
        new=None,
        status: str = "applied",
        detail: str | None = None,
        t_requested: int | None = None,
        t_applied: int | None = None,
        source: str | None = None,
    ) -> dict:
        row = {
            "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "t_requested": t_requested,
            "t_applied": t_applied,
            "experiment": experiment,
            "channel": channel,
            "kind": kind,
            "key": key,
            "old": _text(old),
            "new": _text(new),
            "status": status,
            "detail": detail,
            "source": source,
        }
        with self._lock:
            self.rows.append(row)
            self._write()
        return row

    def table(self) -> pa.Table:
        columns = {k: [r.get(k) for r in self.rows] for k in EVENT_COLUMNS}
        return pa.table(columns, schema=EVENT_SCHEMA)

    def _write(self):
        if self.path is None:
            return
        try:
            tmp = self.path.with_suffix(".parquet.tmp")
            pq.write_table(self.table(), tmp)
            shutil.move(str(tmp), str(self.path))
        except Exception as e:
            logger.error(f"failed to write {self.path}: {e}", exc_info=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.path is None or not self.rows:
            return
        with self._lock:
            self._write()
            try:
                pacsv.write_csv(self.table(), self.path.with_suffix(".csv"))
            except Exception as e:
                logger.error(f"failed to write events csv: {e}", exc_info=True)

    def __len__(self) -> int:
        return len(self.rows)
