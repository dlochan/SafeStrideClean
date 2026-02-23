from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        if list(r.keys()) != fieldnames:
            raise ValueError("All rows must have identical keys")

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_knee_metrics_2d_artifacts(
    *,
    out_dir: str | Path,
    trial_id: str,
    metrics_per_window: List[Dict[str, Any]],
    metrics_summary: Dict[str, Any],
    overwrite: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    d = out_dir / str(trial_id)
    d.mkdir(parents=True, exist_ok=True)

    per_win_path = d / "metrics_per_window.csv"
    if per_win_path.exists() and not overwrite:
        raise FileExistsError(str(per_win_path))

    rows = list(metrics_per_window)
    rows = sorted(rows, key=lambda r: int(r.get("window_index", 0)))
    _write_csv(per_win_path, rows)

    summary_path = d / "metrics_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(str(summary_path))

    summary_path.write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return d
