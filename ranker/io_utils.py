"""I/O helpers: stream-load candidates from JSONL/JSONL.gz and write submission CSV.

Loading is streaming to keep peak memory well within the 16 GB budget even for
the full 100K pool (the file is ~487 MB uncompressed).
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterator


def _open_maybe_gzip(path: Path):
    """Open a file transparently whether it is gzipped or plain text."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield candidate dicts one at a time from a .jsonl or .jsonl.gz file."""
    path = Path(path)
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str | Path) -> list[dict]:
    """Load all candidate records into a list."""
    return list(iter_candidates(path))


def write_submission(rows: list[dict], out_path: str | Path) -> None:
    """Write the ranked submission CSV in the exact format the validator expects.

    `rows` must be ordered by rank and each row must contain
    candidate_id, rank, score, reasoning.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([
                r["candidate_id"],
                int(r["rank"]),
                f"{float(r['score']):.4f}",
                r["reasoning"],
            ])
