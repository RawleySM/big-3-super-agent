from __future__ import annotations

import json
from pathlib import Path

import sqlite3


def load_traces(jsonl_path: Path, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                timestamp REAL,
                trace_id TEXT,
                job_id TEXT,
                phase TEXT,
                payload TEXT
            )
            """
        )
        conn.execute("DELETE FROM traces")
        with jsonl_path.open("r", encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp if line.strip()]
        conn.executemany(
            "INSERT INTO traces (timestamp, trace_id, job_id, phase, payload) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row.get("timestamp"),
                    row.get("trace_id"),
                    row.get("job_id"),
                    row.get("phase"),
                    json.dumps(row.get("payload", {}), ensure_ascii=False),
                )
                for row in rows
            ],
        )
        conn.commit()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert trace JSONL logs to SQLite for Datasette")
    parser.add_argument("jsonl", type=Path, help="Trace JSONL file")
    parser.add_argument("sqlite", type=Path, help="SQLite destination path")
    args = parser.parse_args()
    load_traces(args.jsonl, args.sqlite)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
